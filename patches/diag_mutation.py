"""Diagnostic: can mutation + readout retraining find improvements, or is the search system the problem?

Measures, on the current champion at the current stage (oracle-only readout fit = round 0, which was
the best round in every recent plateau run), evaluated on gate+final task sets (250 eps/family):
  champ     : champion re-fitted with different readout data seeds    -> readout-retraining noise floor
  neutral   : copy + retune only (homeostasis + g_rec search)          -> effect of the retune step itself
  tiny      : +-0.5% edges, retune                                     -> small random permutations
  current   : the production mutation operator (trn.mutate.mutate)    -> what evolution has been trying
  targeted  : rewire the inputs of the 5% least-used mixing neurons    -> readout-informed mutation
Results appended to $OUT (jsonl), one line per trained network.
"""
import json, os, random, sys, time, torch
sys.path.insert(0, "/workspace/TRN")
from trn import evolve as E, mutate as M
from trn.reservoir import Reservoir

E.MAX_ROUNDS = 0
OUT = os.environ.get("OUT", "/workspace/trn_state/diag_mutation.jsonl")
STATE = E.STATE
text, _ = E.corpus()
st = json.load(open(os.path.join(STATE, "state.json")))
stage = st["stage"]
champ_state = torch.load(os.path.join(STATE, "champion.pt"))
champ_heads = champ_state["heads"]


def importance(res, heads):
    """per-neuron share of readout use: sum over heads of normalised |w|*std(feature)."""
    from trn.agent import Runner
    st_ = res.init_state(64)
    g = torch.Generator().manual_seed(5)
    offs = torch.randint(0, len(text) - 400, (64,), generator=g)
    acc, acc2, n = 0, 0, 0
    for t in range(300):
        res.step(st_, torch.stack([text[o + t] for o in offs]).cuda())
        if t >= 100:
            F = res.features(st_).float(); acc = acc + F.sum(0); acc2 = acc2 + (F * F).sum(0); n += F.shape[0]
    sd = (acc2 / n - (acc / n) ** 2).clamp_min(0).sqrt()
    imp = torch.zeros(res.N, device="cuda")
    for k, W in heads.items():
        c = (W[:-1].cuda().abs() * sd[:, None]).pow(2).sum(1)
        c = c / c.sum()
        imp += c[: res.N] + c[res.N:]
    return imp


def rewire_least_used(r, frac, imp, rng):
    mix = torch.nonzero(~r.is_chain).flatten()
    k = int(len(mix) * frac)
    low = mix[imp[mix].argsort()[:k]]
    g = torch.Generator().manual_seed(rng.randrange(2 ** 31))
    K = r.rules["fan_in"]
    for j in low.tolist():
        r.W[:, j] = 0
        src = torch.randint(0, r.N, (K,), generator=g)
        r.W[src.cuda(), j] = (torch.randint(0, 2, (K,), generator=g) * 2 - 1).to(torch.int8).cuda()
    return f"rewired {k} least-used neurons"


def evaluate_big(res, heads):
    a = E.evaluate(res, heads, stage, "gate")
    b = E.evaluate(res, heads, stage, "final")
    n_a, n_b = E.N_GATE, E.N_FINAL
    fams = list(a["families"])
    fam = {f: (a["families"][f] * n_a + b["families"][f] * n_b) / (n_a + n_b) for f in fams}
    lm = (a["lm_acc"] + b["lm_acc"]) / 2
    wl = E.w_lm(stage)
    fit = (1 - wl) * sum(fam.values()) / len(fams) + wl * lm
    succ = [x for x, _ in a["items"][: len(fams) * n_a]] + [x for x, _ in b["items"][: len(fams) * n_b]]
    return fit, fam, lm, succ


def run(arm, rep, seed):
    t0 = time.time()
    res = Reservoir.from_state(champ_state["res"])
    rng = random.Random(seed)
    desc = []
    if arm == "neutral":
        desc.append(f"retuned echo={res.tune(text)}")
    elif arm == "tiny":
        desc += [M.add_edges(res, rng, 0.005), M.del_edges(res, rng, 0.005), f"retuned echo={res.tune(text)}"]
    elif arm == "current":
        desc += M.mutate(res, rng, text, grow=0)
    elif arm == "targeted":
        imp = importance(res, champ_heads)
        desc += [rewire_least_used(res, 0.05, imp, rng), f"retuned echo={res.tune(text)}"]
    heads = E.train_to_plateau(res, stage, seed=seed, tag=f"diag {arm} {rep}")
    fit, fam, lm, succ = evaluate_big(res, heads)
    rec = {"arm": arm, "rep": rep, "seed": seed, "desc": desc, "fitness": round(fit, 4), "lm": round(lm, 4),
           "families": {k: round(v, 3) for k, v in fam.items()}, "succ": succ, "minutes": round((time.time() - t0) / 60, 1)}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(arm, rep, rec["fitness"], rec["minutes"], "min", flush=True)
    del res, heads
    torch.cuda.empty_cache()


plan = [("champ", 0, 9001), ("tiny", 0, 101), ("current", 0, 201), ("targeted", 0, 301), ("neutral", 0, 401),
        ("champ", 1, 9002), ("tiny", 1, 102), ("current", 1, 202), ("targeted", 1, 302),
        ("champ", 2, 9003), ("tiny", 2, 103), ("current", 2, 203), ("targeted", 2, 303), ("neutral", 1, 402)]
done = set()
if os.path.exists(OUT):
    done = {(json.loads(l)["arm"], json.loads(l)["rep"]) for l in open(OUT)}
for arm, rep, seed in plan:
    if (arm, rep) not in done:
        run(arm, rep, seed)
print("DIAG DONE", flush=True)
