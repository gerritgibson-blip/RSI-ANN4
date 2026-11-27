"""Evolution orchestrator (resumable; state in $TRN_STATE, default /workspace/trn_state).

v3 comparison system (2026-11-24; a diagnostic showed readout-retraining noise ~0.008 fitness swamped
mutation effects, so the old system could not see improvements):
  * common random numbers: within an EPOCH every network's readouts are fitted on the SAME training
    episodes (train_seed), and old vs new are scored on the same GATE tasks (paired)
  * the gate set and training seed rotate every EPOCH generations (champion re-fitted) so selection
    cannot over-fit one sample
  * neutral drift: a mutant replaces the champion when it is better on the gate by z >= ACCEPT_Z (0.5),
    so many small real gains can accumulate
  * an ADVANCE (real, reported improvement) needs a paired z >= 1.5 on the FINAL set, which is never
    used for selection, against the last recorded champion
  * mutation operators act at module level (trn.mutate): word-match module, sub-population re-wiring
    with new statistics, rules, growth / neuron deletion within the hardware limit
Readouts are fitted on oracle demonstrations (DAgger rounds never beat round 0 at stage 3 -> MAX_ROUNDS 0).
"""
import copy, json, math, os, random, shutil, sys, time, torch
from .reservoir import Reservoir, encode, VOCAB
from .readout import Ridge, acc_metric, multilabel_metric, auc_metric
from .agent import Runner
from .env import FAMILIES, STAGES, STAGE_NAMES, ACTIONS, MAXPTR
from . import mutate as M

STATE = os.environ.get("TRN_STATE", "/workspace/trn_state")
N0 = os.environ.get("TRN_N0")        # default: as large as the hardware allows
SMOKE = os.environ.get("TRN_SMOKE") == "1"
N_GATE, N_DEV, N_FINAL = (8, 6, 4) if SMOKE else (100, 100, 150)
EP_ROUND0, EP_ROUND = (40, 30) if SMOKE else (1500, 800)       # episodes per active family
MAX_ROUNDS = 0
ACCEPT_Z = 1.0          # neutral-drift acceptance (selection)
RACE_Z = -2.0           # racing: a mutant this far behind the champion after 1/3 of the readout data is dropped
RACE_FRAC = 3           # first chunk = 1/RACE_FRAC of the training episodes
DEV_Z = 0.0             # a gate-passing mutant must also not be worse on the independent dev set (stops downward drift)
ADVANCE_Z = 1.5         # confirmed improvement (reporting), on the never-selected final set
EPOCH = 3 if SMOKE else 10
N_GATE = 8 if SMOKE else 150
B = 256
N_LM = 64
_CORPUS = None
EVAL_MULT = {}        # EVAL2_PATCH: family -> evaluation-task multiplier (fixed per stage, st['eval_mult'])
EVAL_MULT_K = 4
GATE_ALPHA = float(os.environ.get("TRN_GATE_ALPHA", 1.0))   # GATE2_PATCH: shrinkage toward the default mix
EPGUARD_DROP = float(os.environ.get("TRN_EPGUARD_DROP", 0.01))   # EPGUARD_PATCH
TRAIN_EMPH = None     # ALLOC3_PATCH: rate-limited per-family data share (set per epoch by main)
TRAIN_ACC = None      # ALLOC_PATCH: champion gate accuracy per family (set per epoch by main)


def transcripts(split, n_chars, seed):
    """language corpus = oracle transcripts of agentic episodes (task, own actions, tool output),
    so 'inner speech' is trained only on agentic language, not on unrelated text."""
    rng, out, n = random.Random(seed), [], 0
    fams = list(FAMILIES)
    while n < n_chars:
        e = FAMILIES[rng.choice(fams)](random.Random(rng.randrange(10 ** 9)), split)
        obs = e.first
        while True:
            out.append(obs); n += len(obs) + 1
            if e.done:
                break
            a, args = e.oracle()
            obs = e.act(a, args)
    return " ".join(out)


def corpus():
    """(train ids, held-out ids): agent transcripts with train-split names / unseen test-split names."""
    global _CORPUS
    if _CORPUS is None:
        p = os.path.join(STATE, "corpus_agentic.json")
        if not os.path.exists(p):
            json.dump({"train": transcripts("train", 4_000_000, 1), "test": transcripts("test", 400_000, 2)}, open(p, "w"))
        d = json.load(open(p))
        _CORPUS = (torch.tensor(encode(d["train"])), torch.tensor(encode(d["test"])))
    return _CORPUS


def log(msg, **kw):
    e = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "msg": msg, **kw}
    with open(os.path.join(STATE, "events.jsonl"), "a") as f:
        f.write(json.dumps(e) + "\n")
    print(json.dumps(e), flush=True)


def active(stage):
    return [f for s in range(stage + 1) for f in STAGES[s]]


def w_lm(stage):
    return 0.1


def specs_for(fams, n, base, split):
    return [(f, base + 1000 * i + k, split) for k, f in enumerate(fams) for i in range(n)]


def sample_specs(weights, n_per, seed, split="train"):
    rng = random.Random(seed)
    fams = list(weights)
    tot = sum(weights.values())
    n = n_per * len(fams)
    out = [(rng.choices(fams, [weights[f] / tot for f in fams])[0], rng.randrange(10 ** 9) + 10 ** 7, split) for _ in range(n)]
    return out


# ---------------- readout genome: query-key attention ----------------
_ATTN = {}


def attn_cfg(res):
    """Runner attention config from the readout genome in res.rules (None = per-word pointer only)."""
    mix = res.rules.get("attn_mix") or {}       # BIND_PATCH
    bind = res.rules.get("bind_mix") or {}
    if not any(v > 0 for v in list(mix.values()) + list(bind.values())):
        return None
    K = res.rules.get("attn_K", 128)
    key = (res.D, K, str(res.device))
    if key not in _ATTN:
        g = torch.Generator().manual_seed(7 * 1000003 + res.D)
        _ATTN.clear()
        _ATTN[key] = ((torch.randint(0, 3, (res.D, K), generator=g) - 1).float() / res.D ** 0.5).to(res.device)
    return {"R": _ATTN[key], "mix": mix, "bind": bind, "key_id": res.rules.get("key_id", 0), "key_mix": res.rules.get("key_mix"),
            "hops": res.rules.get("hops", 1), "carry": res.rules.get("carry", 0), "note_len": res.rules.get("note_len", 4)}     # REASON_PATCH MEMO2_PATCH


def hop_on(cfg):
    return bool(cfg and any(v > 0 for v in (cfg.get("bind") or {}).values()) and (cfg.get("hops") or 1) >= 2)


def key_dim(cfg):
    from .agent import SIG_S
    return 2 * cfg["R"].shape[1] + (SIG_S if cfg.get("key_id") else 0)


def n_ptr(cfg):
    b = 1 if cfg and any(v > 0 for v in (cfg.get("bind") or {}).values()) else 0
    return MAXPTR * (len(ACTIONS) + b + (1 if cfg and cfg.get("carry") else 0))


def op_weights(hist):
    """directed mutation: operator probabilities follow each operator's recent track record (mean paired z
    of its last 12 uses, shrunk toward 0 with a prior of 3 pseudo-uses)."""
    w = dict(M.OPS)
    for op in w:
        zs = [h["z"] for h in hist if op in h.get("ops", [])][-12:]
        mz = sum(zs) / (len(zs) + 3)
        w[op] = M.OPS[op] * math.exp(max(-2.0, min(2.0, mz)))
    return w


# ---------------- evaluation ----------------
def evaluate(res, heads, stage, which, epoch=0):
    base, n = {"gate": (10_000_000 + 1_000_000 * epoch, N_GATE), "dev": (2_000_000, N_DEV), "final": (3_000_000, N_FINAL)}[which]
    fams = active(stage)
    run = Runner(res, heads, corpus()[0], attn=attn_cfg(res))
    nf = {f: n * int(EVAL_MULT.get(f, 1)) for f in fams}
    specs = [(f, base + 1000 * i + k, "test") for k, f in enumerate(fams) for i in range(nf[f])]
    results, _ = run.run(specs, mode="agent", B=B)
    results.sort(key=lambda r: (r["family"], r["seed"]))
    lm, lm_items = run.lm_eval(corpus()[1], seed={"gate": 1000 + epoch, "dev": 22, "final": 33}[which])
    fam = {f: sum(r["success"] for r in results if r["family"] == f) / nf[f] for f in fams}
    wl = w_lm(stage)
    fitness = (1 - wl) * sum(fam.values()) / len(fams) + wl * lm
    items = [(float(r["success"]), (1 - wl) / (len(fams) * nf[r["family"]])) for r in results] + [(x, wl / len(lm_items)) for x in lm_items]
    return {"fitness": fitness, "families": fam, "lm_acc": lm, "items": items}


def paired_z(new, old):
    d = [a[0] - b[0] for a, b in zip(new["items"], old["items"])]
    w = [a[1] for a in new["items"]]
    delta = sum(wi * di for wi, di in zip(w, d))
    var = sum(wi * wi * (di - delta) ** 2 for wi, di in zip(w, d)) * len(d) / max(1, len(d) - 1)
    return delta, (delta / math.sqrt(var) if var > 0 else (0.0 if delta == 0 else math.copysign(9.9, delta)))


# ---------------- readout training to plateau ----------------
def fit_heads(acc):
    A = len(ACTIONS)
    dec_metric = lambda P, Y: 0.5 * acc_metric(P[:, :A], Y[:, :A]) + 0.5 * acc_metric(P[:, A:A + 4], Y[:, A:A + 4])
    heads, val = {}, {}
    for name, metric in (("lm", acc_metric), ("ptr", auc_metric), ("dec", dec_metric)):
        NP = MAXPTR * len(ACTIONS)
        grp = ([slice(0, NP)] + [slice(c, c + MAXPTR) for c in range(NP, acc[name].C, MAXPTR)]) if (name == "ptr" and acc[name].C > NP) else None
        val[name] = round(acc[name].fit(metric, groups=grp), 4)
        W = acc[name].W.clone()
        for tname, tn in reversed(getattr(acc[name], "tail", [])):   # CARRY2_PATCH: extra heads after the query columns
            heads[tname] = W[:, W.shape[1] - tn:].contiguous()
            W = W[:, : W.shape[1] - tn]
        if name == "dec":
            heads["act"], heads["nargs"] = W[:, :A].contiguous(), W[:, A:A + 4].contiguous()
            if W.shape[1] > A + 4:
                heads["q"] = W[:, A + 4:].contiguous()           # attention query head
        else:
            heads[name] = W
        torch.cuda.empty_cache()
    return heads, val


def train_to_plateau(res, stage, seed, tag="", race=None):
    """race(heads_after_chunk1) -> bool: False stops training early (returns None)."""
    t0 = time.time()
    D = res.D
    cfg = attn_cfg(res)
    acc = {"lm": Ridge(D, VOCAB), "ptr": Ridge(D, n_ptr(cfg)),
           "dec": Ridge(D, len(ACTIONS) + 4 + (key_dim(cfg) if cfg else 0) + (3 if hop_on(cfg) else 0) + ((__import__("trn.agent", fromlist=["carry_cap"]).carry_cap(cfg) + 1) if cfg and cfg.get("carry") else 0))}   # MEMO2_PATCH
    acc["dec"].tail = ([("hop", 3)] if hop_on(cfg) else []) + ([("ncarry", __import__("trn.agent", fromlist=["carry_cap"]).carry_cap(cfg) + 1)] if cfg and cfg.get("carry") else [])   # STAGE5_PATCH MEMO2_PATCH
    fams = active(stage)
    emph = {f: (1.0 + FAMILIES[f].stage) * (1.5 if FAMILIES[f].stage == stage else 1.0) for f in fams}   # agentic + newest skills weigh most
    if TRAIN_ACC:                                     # error-driven allocation: data goes where the failures are
        emph = {f: max(1.0, emph[f] / 3) + 12.0 * (1.0 - TRAIN_ACC.get(f, 0.5)) for f in fams}
    if TRAIN_EMPH and set(TRAIN_EMPH) == set(fams):     # ALLOC3_PATCH
        emph = dict(TRAIN_EMPH)
    run = Runner(res, {}, corpus()[0], attn=cfg)
    # two chunks of oracle data; the readout statistics are additive, so a racing check after chunk 1 costs
    # only one extra solve + gate evaluation, and clear losers skip the remaining 2/3 of the data collection
    n1 = EP_ROUND0 // RACE_FRAC
    run.run(sample_specs(emph, n1, seed), mode="oracle", collect=acc, n_lm=N_LM, B=B, seed=seed)
    if race is not None:
        h1, _ = fit_heads(acc)
        if not race(h1):
            del acc, h1
            torch.cuda.empty_cache()
            log("raced out", tag=tag, N=res.N, minutes=round((time.time() - t0) / 60, 1))
            return None
        del h1
    run.run(sample_specs(emph, EP_ROUND0 - n1, seed + 1), mode="oracle", collect=acc, n_lm=N_LM, B=B, seed=seed + 1)
    heads, val = fit_heads(acc)
    if MAX_ROUNDS == 0:
        del acc
        torch.cuda.empty_cache()
        if cfg is not None and res.rules.get("gate"):        # GATE_PATCH: contextual component gate, fresh episodes
            from .agent import GateFit
            gf = GateFit(cfg["R"].shape[1] + len(ACTIONS) + 1)
            run.heads = heads
            run.run(sample_specs({f: 1.0 for f in fams}, max(4, EP_ROUND0 // int(res.rules.get("gate_ep") or 8)), seed + 2), mode="gate", collect={"gate": gf}, n_lm=0, B=B, seed=seed + 2)
            if int(res.rules.get("gate") or 0) == 2:      # GATE2_PATCH: prior-centred fit
                heads["gate"] = gf.solve(float(res.rules.get("gate_alpha") or GATE_ALPHA), w0=run.gate_prior(gf.ds))   # GALPHA_PATCH
            else:
                heads["gate"] = gf.solve()
            val["gate_rows"] = gf.n
        log("readouts fitted", tag=tag, N=res.N, val=val, minutes=round((time.time() - t0) / 60, 1))
        return heads
    dev = evaluate(res, heads, stage, "dev")
    best = (dev["fitness"], heads, dev, 0)
    hist = [round(dev["fitness"], 4)]
    patience = 0
    for r in range(1, MAX_ROUNDS + 1):
        # failure-driven: sample training tasks where the agent fails on dev
        w = {f: emph[f] * (1.05 - dev["families"][f]) for f in fams}
        run.heads = heads
        run.run(sample_specs(w, EP_ROUND, seed + r), mode="dagger", collect=acc, beta=0.3, n_lm=N_LM, B=B, seed=seed + r)
        heads, val = fit_heads(acc)
        dev = evaluate(res, heads, stage, "dev")
        hist.append(round(dev["fitness"], 4))
        if dev["fitness"] > best[0] + 0.005:
            best, patience = (dev["fitness"], heads, dev, r), 0
        else:
            patience += 1
            if patience >= 2:
                break
    del acc
    torch.cuda.empty_cache()
    log("readouts plateaued", tag=tag, N=res.N, rounds=len(hist), dev_curve=hist, best_round=best[3], val=val,
        minutes=round((time.time() - t0) / 60, 1))
    return best[1]


# ---------------- hardware limit ----------------
def fits_hardware(N, H=0, D=None):
    """3 fp32 Gram matrices on the GPU during collection (solves work on a host-parked copy) + 1/12 margin, ternary W, validation rows, flush buffers,
    word-match module state (B x H x H fast weights + gather copy)."""
    D = D or 2 * N + 3 * H     # 3H: module r0, r1 + optional bag-of-words features
    need = 13 * D * D + N * N + 3 * 20000 * D + 3 * 2 ** 29 + 2.5e9 + 3 * 4 * B * H * H
    return need < 0.85 * torch.cuda.get_device_properties(0).total_memory


def max_neurons():
    N = 1024
    while fits_hardware(N + 256):
        N += 256
    return N


# ---------------- persistence ----------------
def save_net(path, res, heads):
    torch.save({"res": res.state_dict(), "heads": {k: v.cpu() for k, v in heads.items()}}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_net(path):
    d = torch.load(path)
    return Reservoir.from_state(d["res"]), {k: v.cuda() for k, v in d["heads"].items()}


def summary(ev):
    return {"fitness": round(ev["fitness"], 4), "lm_acc": round(ev["lm_acc"], 4), "families": {k: round(v, 3) for k, v in ev["families"].items()}}


def main():
    os.makedirs(STATE, exist_ok=True)
    sp = os.path.join(STATE, "state.json")
    st = json.load(open(sp)) if os.path.exists(sp) else {"readout_version": 3, "gen": 0, "stage": 0, "rejects": 0, "history": [], "best": None}
    save = lambda: json.dump(st, open(sp, "w"), indent=1)
    champ_path = os.path.join(STATE, "champion.pt")
    text, _ = corpus()
    fits = lambda N, H, D=None: fits_hardware(N, H, D)

    if not os.path.exists(champ_path):
        res = Reservoir(int(N0) if N0 else max_neurons(), seed=1)
        echo = res.tune(text)
        log("initial reservoir", N=res.N, edges=res.n_edges(), g_rec=res.g_rec, echo=echo,
            delay_neurons=int(res.is_chain.sum()), input_nodes=int(res.is_in.sum()))
        st.update(readout_version=3, epoch=0, epoch_gens=0, train_seed=101, record=None, force_ops=["module"])
        heads = train_to_plateau(res, st["stage"], seed=st["train_seed"], tag="initial")
        save_net(champ_path, res, heads)
    res, heads = load_net(champ_path)

    def rebaseline(kind):
        nonlocal heads
        def rec1(h1):
            st["best_items1"] = evaluate(res, h1, st["stage"], "gate", st["epoch"])["items"]
            return True
        heads = train_to_plateau(res, st["stage"], seed=st["train_seed"], tag=f"champion epoch {st['epoch']}", race=rec1)
        save_net(champ_path, res, heads)
        ev = evaluate(res, heads, st["stage"], "gate", st["epoch"])
        st["best"] = summary(ev); st["best_items"] = ev["items"]
        st["best_dev_items"] = evaluate(res, heads, st["stage"], "dev")["items"]
        if kind:
            save_net(os.path.join(STATE, "champion_record.pt"), res, heads)
            fin = evaluate(res, heads, st["stage"], "final")
            st["record"] = summary(fin); st["record_items"] = fin["items"]
            log("ADVANCE", kind=kind, N=res.N, stage=st["stage"], stage_name=STAGE_NAMES[st["stage"]], module=module_desc(res),
                test_set="final", **summary(fin))
        save()

    global TRAIN_ACC, TRAIN_EMPH, EVAL_MULT
    EVAL_MULT = st.get("eval_mult") or {}             # EVALFIX_PATCH: before any rebaseline below
    TRAIN_ACC = st.get("train_acc")
    TRAIN_EMPH = st.get("train_emph")                 # ALLOC3_PATCH
    if st.get("readout_version", 1) != 3:
        st.update(readout_version=3, epoch=0, epoch_gens=0, train_seed=101, force_ops=["module"])
        log("comparison system v3 (common random numbers, rotating gate, neutral drift, module mutations); research arg-order fix")
        rebaseline("re-baseline: new comparison system + research fix")
    if st["best"] is None:
        rebaseline("initial champion")
    if "best_items1" not in st:
        log("racing enabled: mutants scored after 1/3 of readout data; dropped if z <", race_z=RACE_Z)
        rebaseline(None)

    EVAL_MULT = st.get("eval_mult") or {}
    if st.get("eval_v") != 2 and st.get("record"):        # EVAL2_PATCH migration
        EVAL_MULT = st["eval_mult"] = {f: EVAL_MULT_K for f, a in st["record"]["families"].items() if a < 0.99}
        rr, rh = load_net(os.path.join(STATE, "champion_record.pt"))
        fin = evaluate(rr, rh, st["stage"], "final")
        old = st["record"]["fitness"]
        st["record"] = summary(fin); st["record_items"] = fin["items"]
        del rr, rh; torch.cuda.empty_cache()
        log("stratified evaluation", eval_mult=EVAL_MULT, record_remeasured=round(fin["fitness"], 4), record_before=old,
            items=len(fin["items"]))
        rebaseline(None)
        fin = evaluate(res, heads, st["stage"], "final")
        fd, fz = paired_z(fin, {"items": st["record_items"]})
        log("protocol check", final=round(fin["fitness"], 4), final_delta=round(fd, 4), final_z=round(fz, 2), **summary(fin))
        if fd > 0 and fz >= ADVANCE_Z:
            prev = st["record"]["fitness"]
            st["record"] = summary(fin); st["record_items"] = fin["items"]
            save_net(os.path.join(STATE, "champion_record.pt"), res, heads)
            log("ADVANCE", kind="champion beat record (re-measured, stratified final set)", gen=st["gen"], N=res.N,
                module=module_desc(res), z=round(fz, 2), previous=prev, test_set="final",
                mutation=["accepted since gen 557: subpop x2, readout attention remember x2.0"], **summary(fin))
        st["eval_v"] = 2
        save()
    if "best_dev_items" not in st:
        st["best_dev_items"] = evaluate(res, heads, st["stage"], "dev")["items"]
        log("dev-set agreement enabled: accepted mutants must also not be worse on the dev set", dev_z=DEV_Z)
        save()

    while not os.path.exists(os.path.join(STATE, "STOP")):
        # ---- curriculum: advance when competent at the current stage ----
        cur = [st["best"]["families"][f] for f in STAGES[st["stage"]]]
        competent = sum(cur) / len(cur) >= 0.8 and min(cur) >= 0.6
        stuck = st.get("stage_gens", 0) >= 8
        if st["stage"] < max(STAGES) and (competent or stuck):
            st["stage"] += 1; st["stage_gens"] = 0
            log("stage change", reason="competent" if competent else "plateaued at this stage for 8 generations")
            rebaseline("curriculum stage")
        # ---- epoch rotation: new training sample + new gate set, champion re-fitted ----
        if st["epoch_gens"] >= EPOCH:
            st["epoch"] += 1; st["epoch_gens"] = 0; st["train_seed"] = 101 + 7919 * st["epoch"]
            if st.get("alloc") and not st.pop("bad_epoch", False):   # ALLOC2_PATCH / EPGUARD_PATCH
                st.setdefault("acc_hist", []).append(dict(st["best"]["families"]))
                st["acc_hist"] = st["acc_hist"][-4:]
                st["train_acc"] = {f: min(h.get(f, 1.0) for h in st["acc_hist"]) for f in st["best"]["families"]}
            TRAIN_ACC = st.get("train_acc")
            if TRAIN_ACC:                             # ALLOC3_PATCH: share may fall by at most 40% per epoch
                fams_ = active(st["stage"])
                tgt = {f: (max(1.0, (1.0 + FAMILIES[f].stage) * (1.5 if FAMILIES[f].stage == st["stage"] else 1.0) / 3)
                           + 12.0 * (1.0 - TRAIN_ACC.get(f, 0.5))) for f in fams_}
                tt = sum(tgt.values()); tgt = {f: v / tt for f, v in tgt.items()}
                prev = st.get("train_emph") or {}
                sh = {f: max(tgt[f], 0.85 * prev.get(f, 0.0)) for f in fams_}   # ALLOC4_PATCH: floor 0.85
                ss = sum(sh.values())
                st["train_emph"] = TRAIN_EMPH = {f: round(v / ss, 5) for f, v in sh.items()}
            log("epoch", epoch=st["epoch"], train_acc=TRAIN_ACC, train_share=TRAIN_EMPH)
            rebaseline(None)
            ref_ = (st.get("record") or {}).get("fitness")          # EPGUARD_PATCH: degenerate training sample?
            tries_ = []
            while ref_ and st["best"]["fitness"] < ref_ - EPGUARD_DROP and len(tries_) < 2:
                tries_.append((st["best"]["fitness"], st["train_seed"]))
                log("degenerate training sample", epoch=st["epoch"], train_seed=st["train_seed"],
                    gate_fitness=st["best"]["fitness"], record=ref_, families=st["best"]["families"])
                st["train_seed"] += 1_000_003
                rebaseline(None)
            if tries_ and st["best"]["fitness"] < max(tries_)[0]:   # every retry worse: go back to the best seed
                st["train_seed"] = max(tries_)[1]
                rebaseline(None)
            if st.pop("advance_check", False):        # one-shot: a training-protocol change judged on the final set
                fin = evaluate(res, heads, st["stage"], "final")
                fd, fz = paired_z(fin, {"items": st["record_items"]})
                log("protocol check", final=round(fin["fitness"], 4), final_delta=round(fd, 4), final_z=round(fz, 2), **summary(fin))
                if fd > 0 and fz >= ADVANCE_Z:
                    prev = st["record"]["fitness"]
                    st["record"] = summary(fin); st["record_items"] = fin["items"]
                    save_net(os.path.join(STATE, "champion_record.pt"), res, heads)
                    log("ADVANCE", kind="training-data allocation", gen=st["gen"], N=res.N, module=module_desc(res), z=round(fz, 2),
                        previous=prev, test_set="final", mutation=["error-driven training allocation"], **summary(fin))
                save()

        st["gen"] += 1; st["epoch_gens"] += 1
        st["stage_gens"] = st.get("stage_gens", 0) + 1
        g = st["gen"]
        rng = random.Random(1000 + g)
        dn = max(512, int(res.N * 0.25) // 256 * 256)
        mut = Reservoir.from_state(res.state_dict())
        t0 = time.time()
        desc = M.mutate(mut, rng, text, grow=dn, fits=fits, force=st.pop("force_ops", None), weights=op_weights(st["history"]))
        race = {}
        def judge(h1):
            race["ev"] = evaluate(mut, h1, st["stage"], "gate", st["epoch"])
            race["items"] = race["ev"]["items"]
            race["z"] = paired_z(race["ev"], {"items": st["best_items1"]})[1]
            return race["z"] >= RACE_Z
        if set(M.mutate.last_ops) == {"readout"} and "q" in heads and not mut.rules.get("gate"):   # GATE3_PATCH
            # readout genome only: network and training sample are identical to the champion's, so the fitted
            # readouts would be identical too -> reuse them and skip data collection entirely
            mheads, race["z"] = heads, float("nan")
        else:
            mheads = train_to_plateau(mut, st["stage"], seed=st["train_seed"], tag=f"gen {g} mutant", race=judge)
        ev = race["ev"] if mheads is None else evaluate(mut, mheads, st["stage"], "gate", st["epoch"])
        delta, z = paired_z(ev, {"items": st["best_items" if mheads is not None else "best_items1"]})
        thr = st.pop("force_accept_z", None)          # one-shot override (operator-directed change, e.g. growth)
        accept = mheads is not None and ((delta > 0 and z >= ACCEPT_Z) if thr is None else z >= thr)
        dz = None
        if accept:
            dv = evaluate(mut, mheads, st["stage"], "dev")
            dz = paired_z(dv, {"items": st["best_dev_items"]})[1]
            if thr is None and dz < DEV_Z:
                accept = False
        rec = {"gen": g, "epoch": st["epoch"], "ops": M.mutate.last_ops, "attn": mut.rules.get("attn_mix"), "mutation": desc, "N": mut.N, "module": module_desc(mut), "fitness": round(ev["fitness"], 4),
               "delta": round(delta, 4), "z": round(z, 2), "z1": (None if race["z"] != race["z"] else round(race["z"], 2)), "raced_out": mheads is None, "dev_z": None if dz is None else round(dz, 2), "accepted": accept, "families": summary(ev)["families"],
               "lm_acc": round(ev["lm_acc"], 4), "minutes": round((time.time() - t0) / 60, 1)}
        if accept:
            if "items" in race:
                st["best_items1"] = race["items"]
            res, heads = mut, mheads
            save_net(champ_path, res, heads)
            st["best"] = summary(ev); st["best_items"] = ev["items"]; st["rejects"] = 0
            st["best_dev_items"] = dv["items"]
            fin = evaluate(res, heads, st["stage"], "final")
            fd, fz = paired_z(fin, {"items": st["record_items"]})
            rec.update(final=round(fin["fitness"], 4), final_delta=round(fd, 4), final_z=round(fz, 2))
            log("accepted", **rec)
            if fd > 0 and fz >= ADVANCE_Z:
                prev = st["record"]["fitness"]
                st["record"] = summary(fin); st["record_items"] = fin["items"]
                save_net(os.path.join(STATE, "champion_record.pt"), res, heads)
                log("ADVANCE", kind="mutant beat recorded champion", gen=g, N=res.N, module=module_desc(res), z=round(fz, 2),
                    previous=prev, test_set="final", mutation=desc, **summary(fin))
        else:
            st["rejects"] += 1
            log("rejected", **rec)
        st["history"].append(rec)
        save()
        del mut, mheads
        torch.cuda.empty_cache()
    log("stopped")


def module_desc(res):
    m = res.mod
    return None if m is None else {"H": m["H"], "decay": m["decay"], "gm": m["gm"], "fan": m["fan"], "ctx": m.get("ctx_pool"), "bow": m.get("bow", 0)}


if __name__ == "__main__":
    main()
