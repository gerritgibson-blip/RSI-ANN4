"""Auxiliary mutation script: evolutionary permutations of the reservoir super-structure.

  python -m trn.mutate <in.pt> <out.pt> [--seed S] [--grow N]

Operators (a random subset is applied, each with a random magnitude):
  add_edges      add a random number of new random ternary synapses
  del_edges      delete a random number of existing synapses
  add_neurons    grow the network (new mixing neurons, new input nodes, new delay lines)
  leak           re-draw the leak timescale of a random subset of neurons
  reroute        cut a delay line and re-attach the tail to another line (acyclic check)
  rules          mutate network rules: think ticks, echo target, firing target, trace decay, input gain
  module         add / re-parameterise / remove the word-match module (content-addressable fast memory)
  subpop         re-draw a whole sub-population of mixing neurons with NEW statistics (fan-in, sign
                 balance, leak range) -- changes what the network computes, not just which random edges it has
  del_neurons    delete random mixing neurons (also used to make room for modules within the hardware limit)
  readout        readout architecture genome: per-action weight of query-key ATTENTION argument selection
                 (vs the per-word pointer); the network itself is unchanged, so no retune
After any mutation the network is re-tuned (homeostasis + echo search) so activity still echoes and
passively dies out; readouts must then be retrained from scratch (features changed).
"""
import argparse, math, random, torch
from .reservoir import Reservoir, VOCAB


def add_edges(r, rng, frac):
    cols = torch.nonzero(~r.is_chain).flatten()
    n = int(r.n_edges() * frac)
    i = torch.randint(0, r.N, (n,), device=r.device)
    j = cols[torch.randint(0, len(cols), (n,), device=r.device)]
    r.W[i, j] = (torch.randint(0, 2, (n,), device=r.device) * 2 - 1).to(torch.int8)
    return f"+{n} edges"


def del_edges(r, rng, frac):
    nz = torch.nonzero(r.W)
    n = int(len(nz) * frac)
    pick = nz[torch.randperm(len(nz), device=r.device)[:n]]
    r.W[pick[:, 0], pick[:, 1]] = 0
    return f"-{n} edges"


def add_neurons(r, rng, n, chain_frac=0.5, in_frac=0.25):
    n = max(256, n // 256 * 256)
    N0, N1 = r.N, r.N + n
    g = torch.Generator().manual_seed(rng.randrange(2 ** 31))
    W = torch.zeros(N1, N1, dtype=torch.int8, device=r.device)
    W[:N0, :N0] = r.W
    K = r.rules["fan_in"]
    new = torch.arange(N0, N1)
    is_in_new = torch.rand(n, generator=g) < in_frac
    n_chain = int(n * chain_frac)
    body_pool = new[~is_in_new]
    chain_new = body_pool[torch.randperm(len(body_pool), generator=g)[:n_chain]]
    # incoming synapses for new non-chain neurons; outgoing synapses from new neurons into old mixing neurons
    mix_new = torch.tensor([x for x in new.tolist() if x not in set(chain_new.tolist())], dtype=torch.long)
    for j in mix_new.tolist():
        src = torch.randint(0, N1, (K,), generator=g)
        W[src, j] = (torch.randint(0, 2, (K,), generator=g) * 2 - 1).to(torch.int8).to(r.device)
    old_mix = torch.nonzero(~r.is_chain).flatten().cpu()
    for i in new.tolist():
        dst = old_mix[torch.randint(0, len(old_mix), (K // 2,), generator=g)]
        W[i, dst] = (torch.randint(0, 2, (K // 2,), generator=g) * 2 - 1).to(torch.int8).to(r.device)
    r.W, r.N = W, N1
    if r.mod:
        r.mod["Wm"] = torch.cat([r.mod["Wm"], torch.zeros(r.mod["H"], n, dtype=torch.int8, device=r.device)], 1)
    r.k = torch.cat([r.k, torch.randint(1, 9, (n,), generator=g).to(torch.int32).to(r.device)])
    r.theta = torch.cat([r.theta, torch.full((n,), int(r.theta.float().median()), dtype=torch.int32, device=r.device)])
    r.is_in = torch.cat([r.is_in, is_in_new.to(r.device)])
    r.pred = torch.where(r.pred == N0, torch.full_like(r.pred, N1), r.pred)
    r.pred = torch.cat([r.pred, torch.full((n,), N1, dtype=torch.long, device=r.device)])
    r.is_chain = torch.cat([r.is_chain, torch.zeros(n, dtype=torch.bool, device=r.device)])
    # new delay lines start at new input nodes
    heads = new[is_in_new].tolist()
    body = chain_new.tolist()
    rng.shuffle(heads)
    while body and heads:
        L = rng.randint(8, r.rules["chain_max"])
        seg, body = body[:L], body[L:]
        prev = heads.pop()
        for x in seg:
            r.pred[x] = prev; r.is_chain[x] = True; r.k[x] = 1; prev = x
    r.W[:, r.is_chain] = 0
    # input projections onto new input nodes
    Win = torch.zeros(VOCAB, N1, dtype=torch.int8, device=r.device)
    Win[:, :N0] = r.Win
    newin = new[is_in_new]
    fan = max(1, int(len(newin) * r.in_fan / max(1, int(r.is_in[:N0].sum()))))
    for t in range(1, VOCAB):
        sel = newin[torch.randperm(len(newin), generator=g)[:fan]]
        Win[t, sel.to(r.device)] = (torch.randint(0, 2, (len(sel),), generator=g) * 2 - 1).to(torch.int8).to(r.device)
    r.Win = Win
    r.in_fan += fan
    r.to(r.device)
    return f"+{n} neurons (N={N1}, {len(chain_new)} in new delay lines)"


def leak(r, rng, frac):
    m = (torch.rand(r.N, device=r.device) < frac) & ~r.is_chain
    r.k[m] = torch.randint(1, 11, (int(m.sum()),), device=r.device).to(torch.int32)
    return f"leak redraw {int(m.sum())}"


def _ancestors(pred, i, N):
    seen = set()
    while i != N and i not in seen:
        seen.add(i); i = int(pred[i])
    return seen


def reroute(r, rng, n):
    pred = r.pred.cpu()
    chain = torch.nonzero(r.is_chain).flatten().tolist()
    done = 0
    for _ in range(n * 4):
        if done >= n:
            break
        j = rng.choice(chain)
        i = rng.choice(chain)
        if i == j or j in _ancestors(pred, i, r.N):
            continue                           # would create a cycle (never-dying echo)
        if len(_ancestors(pred, i, r.N)) + 1 > 2 * r.rules["chain_max"]:
            continue
        pred[j] = i; done += 1
    r.pred = pred.to(r.device)
    return f"reroute {done} delay links"


def rules(r, rng):
    k = rng.choice(["think", "echo_target", "target_rate", "trace_shift", "g_in", "chain_max", "g_fix", "g_fix", "chain_vq"])
    R = r.rules
    if k == "think":
        R["think"] = min(10, max(1, R["think"] + rng.choice([-1, 1])))
    elif k == "echo_target":
        R["echo_target"] = int(min(600, max(60, R["echo_target"] * rng.choice([0.75, 1.33]))))
    elif k == "target_rate":
        R["target_rate"] = min(0.2, max(0.01, R["target_rate"] * rng.choice([0.7, 1.4])))
    elif k == "trace_shift":
        R["trace_shift"] = min(6, max(1, R["trace_shift"] + rng.choice([-1, 1])))
    elif k == "g_in":
        r.g_in = int(min(600, max(40, r.g_in * rng.choice([0.75, 1.33]))))
    elif k == "chain_max":
        R["chain_max"] = int(min(512, max(16, R["chain_max"] * rng.choice([0.75, 1.33]))))
    elif k == "chain_vq":  # expose relay-neuron membranes to the readout (1) or only their traces (0)
        R["chain_vq"] = 1 - R.get("chain_vq", 1)
    elif k == "g_fix":     # recurrent gain as a gene (None = derive it from the echo target)
        R["g_fix"] = rng.choice([x for x in (None, 0, 1, 2, 4, 8, 16) if x != R.get("g_fix")])
    return f"rule {k} -> {R.get(k, r.g_in)}"


def del_neurons(r, rng, n):
    """delete n random mixing neurons (never delay-line or input neurons, so delay lines stay intact)."""
    n = max(256, n // 256 * 256)
    mix = torch.nonzero(~r.is_chain & ~r.is_in).flatten().cpu()
    used = set(r.mod["ctx"].tolist()) if r.mod else set()
    g = torch.Generator().manual_seed(rng.randrange(2 ** 31))
    if len(mix) < n + 256:
        return f"del_neurons skipped (only {len(mix)} deletable)"
    drop = mix[torch.randperm(len(mix), generator=g)[:n]]
    keep = torch.ones(r.N, dtype=torch.bool)
    keep[drop] = False
    newidx = torch.full((r.N + 1,), -1, dtype=torch.long)
    newidx[torch.nonzero(keep).flatten()] = torch.arange(int(keep.sum()))
    N1 = int(keep.sum())
    newidx[r.N] = N1
    kd = keep.to(r.device)
    r.W = r.W[kd][:, kd].contiguous()
    r.k, r.theta, r.is_in, r.is_chain = r.k[kd], r.theta[kd], r.is_in[kd], r.is_chain[kd]
    r.pred = newidx.to(r.device)[r.pred[kd]]
    r.Win = r.Win[:, kd].contiguous()
    r.N = N1
    if r.mod:
        c = newidx[r.mod["ctx"].cpu()]
        cand = torch.nonzero(~r.is_chain.cpu()).flatten()
        bad = c < 0
        c[bad] = cand[torch.randint(0, len(cand), (int(bad.sum()),), generator=g)]
        r.mod["ctx"] = c
        r.mod["Wm"] = r.mod["Wm"][:, kd].contiguous()
    r.to(r.device)
    return f"-{n} neurons (N={N1})"


def module(r, rng):
    if r.mod is None:
        H, d = rng.choice([256, 384, 512]), rng.choice([4, 5, 6, 7])
        r.add_module(H=H, decay=d, seed=rng.randrange(2 ** 31))
        return f"add word-match module H={H} decay={d}"
    m = r.mod
    k = rng.choice(["decay", "gm", "ctx", "H", "fan", "ctx_pool", "remove"] if rng.random() < 0.1 else ["decay", "gm", "ctx", "H", "fan", "ctx_pool", "bow", "bow"])
    if k == "remove":
        r.mod = None
        return "remove word-match module"
    if k == "decay":
        m["decay"] = min(9, max(2, m["decay"] + rng.choice([-1, 1])))
        return f"module decay -> {m['decay']}"
    if k == "bow":        # bag-of-recent-words features: off, or a word-decay shift (memory of ~2^bow words)
        m["bow"] = rng.choice([x for x in (0, 2, 3, 4, 5, 6, 7, 8) if x != m.get("bow", 0)])   # BOW3_PATCH
        return f"module bag-of-words -> {m['bow']}"
    if k == "gm":
        m["gm"] = int(min(64, max(0, round(m["gm"] * rng.choice([0.5, 2]) + rng.choice([0, 1])))))
        return f"module injection gain -> {m['gm']}"
    H = m["H"] if k != "H" else min(1024, max(128, m["H"] + rng.choice([-128, 128])))
    fan = m["fan"] if k != "fan" else min(64, max(1, m["fan"] * rng.choice([0.5, 2])))
    pool = m.get("ctx_pool", "mixing")
    if k == "ctx_pool":
        pool = "chain" if pool == "mixing" else "mixing"       # context snapshot from delay lines (ordered history) or mixing pool
    keep = {x: m[x] for x in ("decay", "gm")}
    r.add_module(H=H, decay=keep["decay"], gm=keep["gm"], fan=int(fan), seed=rng.randrange(2 ** 31), ctx_pool=pool, bow=m.get("bow", 0))   # re-drawn ctx / hash / projection
    return f"module redraw ({k}) H={H} fan={int(fan)} ctx={pool}"


ATTN_ACTIONS = ["answer", "open", "lookup", "install", "remember", "recall", "search", "calc"]


def readout(r, rng, spec=None):
    mix = dict(r.rules.get("attn_mix") or {})
    if spec:
        a, w = spec
    else:
        a = rng.choice(ATTN_ACTIONS)
        cur = float(mix.get(a, 0.0))
        w = rng.choice([x for x in (0.0, 0.5, 1.0, 2.0, 4.0) if x != cur])  # never a no-op
    mix[a] = float(w)
    r.rules["attn_mix"] = mix
    return f"readout attention {a} x{w}"


def bind(r, rng, spec=None):
    """BIND_PATCH: readout genes of the binding attention (changes head shapes -> not a reuse-able readout op)."""
    if spec:
        k, a, w = spec
    else:
        k = rng.choice(["bind", "bind", "key_id", "key_mix", "key_mix", "ptr_all", "gate", "hops", "carry", "occ", "bmask", "sep_reset", "dedup", "occ_rec", "note_len"])   # DEDUP_PATCH OCCREC_PATCH MEMO_PATCH   # OCC_PATCH BMASK_PATCH SEPRESET_PATCH
        a = rng.choice(ATTN_ACTIONS) if k in ("bind", "key_mix") else None
    if k == "note_len":                                       # MEMO_PATCH
        L = [4, 6, 8]
        cur = int(r.rules.get("note_len") or 4)
        i = L.index(cur) if cur in L else 0
        r.rules["note_len"] = int(w) if spec else L[max(0, min(len(L) - 1, i + rng.choice([-1, 1])))]
        return f"note length -> {r.rules['note_len']}"
    if k == "occ_rec":                                        # OCCREC_PATCH
        L = [0.0, 0.25, 0.5, 1.0, 2.0]
        cur = float(r.rules.get("occ_rec") or 0.0)
        i = min(range(len(L)), key=lambda j: abs(L[j] - cur))
        r.rules["occ_rec"] = float(w) if spec else L[max(0, min(len(L) - 1, i + rng.choice([-1, 1])))]
        return f"recency-aware occurrence pooling -> {r.rules['occ_rec']}"
    if k == "dedup":
        r.rules["dedup"] = int(w) if spec else (0 if r.rules.get("dedup") else 1)
        return f"argument copy distinct words -> {r.rules['dedup']}"
    if k == "sep_reset":
        r.rules["sep_reset"] = int(w) if spec else (0 if r.rules.get("sep_reset") else 1)
        return f"episode-start memory reset -> {r.rules['sep_reset']}"
    if k == "bmask":
        r.rules["bmask"] = int(w) if spec else (0 if r.rules.get("bmask") else 1)
        return f"readout bridge needs a repeated word -> {r.rules['bmask']}"
    if k == "occ":
        cur = int(r.rules.get("occ") or 0)
        r.rules["occ"] = int(w) if spec else {0: 1, 1: 3, 3: 2, 2: 0}[cur]
        return f"readout occurrence linking -> {r.rules['occ']}"
    if k in ("hops", "carry"):                                # REASON_PATCH
        cur = int(r.rules.get(k) or (1 if k == "hops" else 0))
        new = int(w) if spec else ({1: 2, 2: 1} if k == "hops" else {0: 1, 1: 2, 2: 3, 3: 2})[cur]   # STAGE5_PATCH: carry 2 = args too; CARRY4_PATCH: 3 = text-ordered
        r.rules[k] = new
        return f"readout {'binding depth' if k == 'hops' else 'carried note'} -> {new}"
    if k == "gate_alpha" or (k == "gate" and not spec and r.rules.get("gate") == 2 and rng.random() < 0.5):   # GALPHA_PATCH
        L = [0.03, 0.1, 0.3, 1.0, 3.0]
        cur = float(r.rules.get("gate_alpha") or 1.0)
        i = min(range(len(L)), key=lambda j: abs(L[j] - cur))
        r.rules["gate_alpha"] = float(w) if (spec and k == "gate_alpha") else L[max(0, min(len(L) - 1, i + rng.choice([-1, 1])))]
        return f"readout gate shrinkage -> {r.rules['gate_alpha']}"
    if k == "gate":                                           # GATE_PATCH
        r.rules["gate"] = 0 if r.rules.get("gate") else 2      # GATE2_PATCH: v3 gate
        return f"readout contextual gate -> {r.rules['gate']}"
    if k == "ptr_all":
        r.rules["ptr_all"] = 0 if r.rules.get("ptr_all") else 1
        return f"readout pointer targets all occurrences -> {r.rules['ptr_all']}"
    if k == "key_mix":
        mix = dict(r.rules.get("key_mix") or {"answer": 1.0})
        cur = float(mix.get(a, 0.0))
        w = w if spec else rng.choice([x for x in (0.0, 0.5, 1.0, 2.0) if x != cur])
        mix[a] = float(w)
        r.rules["key_mix"] = mix
        return f"readout key identity for {a} x{w}"
    if k == "key_id":
        cur = float(r.rules.get("key_id", 0) or 0)
        w = w if spec else rng.choice([x for x in (0.0, 0.5, 1.0, 2.0) if x != cur])
        r.rules["key_id"] = float(w)
        return f"readout key identity x{w}"
    mix = dict(r.rules.get("bind_mix") or {})
    cur = float(mix.get(a, 0.0))
    w = w if spec else rng.choice([x for x in (0.0, 0.5, 1.0, 2.0, 4.0) if x != cur])
    mix[a] = float(w)
    r.rules["bind_mix"] = mix
    return f"readout bridge attention {a} x{w}"


def subpop(r, rng):
    """a random sub-population of mixing neurons gets new incoming wiring drawn from different statistics."""
    frac = rng.uniform(0.05, 0.3)
    mix = torch.nonzero(~r.is_chain).flatten()
    sel = mix[torch.randperm(len(mix), device=r.device)[: int(len(mix) * frac)]]
    K = int(max(4, min(512, r.rules["fan_in"] * rng.choice([0.25, 0.5, 2, 4]))))
    pexc = rng.choice([0.3, 0.5, 0.7])                       # sign balance (excitatory fraction)
    src_pool = rng.choice(["all", "chain", "mixing"])          # listen mostly to delay lines / to the mixing pool
    pool = {"all": torch.arange(r.N, device=r.device), "chain": torch.nonzero(r.is_chain).flatten(), "mixing": mix}[src_pool]
    r.W[:, sel] = 0
    n = len(sel)
    src = pool[torch.randint(0, len(pool), (K, n), device=r.device)]
    sgn = torch.where(torch.rand(K, n, device=r.device) < pexc, 1, -1).to(torch.int8)
    r.W[src, sel.expand(K, n)] = sgn
    lo = rng.choice([1, 2, 4]); hi = lo + rng.choice([2, 4, 6])
    r.k[sel] = torch.randint(lo, hi + 1, (n,), device=r.device).to(torch.int32)
    return f"subpop {n} neurons: fan_in {K} from {src_pool}, exc {pexc}, leak k {lo}-{hi}"


OPS = {"module": 3, "subpop": 3, "rules": 2, "edges": 1, "leak": 1, "reroute": 1, "del_neurons": 0.5, "grow": 1, "readout": 2, "bind": 1.5}


def mutate(r, rng, corpus_ids, grow=0, fits=None, force=None, weights=None):
    """apply 1-2 operators (weighted toward module-level changes) in place, keep within the hardware limit,
    retune. `fits(r_N, H)` -> bool is the hardware-limit predicate. Returns a description list."""
    log = []
    fits = fits or (lambda N, H, D=None: True)
    W = weights or OPS
    ops = list(force or [])
    while not ops:
        ops = list({rng.choices(list(W), list(W.values()))[0] for _ in range(rng.choice([1, 1, 2]))})
        if "grow" in ops and not (grow and fits(r.N + grow, r.mod["H"] if r.mod else 0, r.D + 2 * grow)):
            ops.remove("grow")
        if "edges" in ops and r.g_rec == 0 and r.rules.get("g_fix") == 0:   # GREC0_PATCH: W is unused -> no-op
            ops.remove("edges")
        if "del_neurons" in ops and int((~r.is_chain & ~r.is_in).sum()) < 512:   # nothing deletable: skip, don't waste a generation
            ops.remove("del_neurons")
    for op in ops:
        if op.startswith("readout"):
            spec = None
            if ":" in op:                               # forced form readout:<action>=<weight>
                a, w = op.split(":", 1)[1].split("=")
                spec = (a, float(w))
            log.append(readout(r, rng, spec))
        elif op.startswith("bind"):                  # forced forms bind:<action>=<w>, bind:key_id=<w>
            spec = None
            if ":" in op:
                a, w = op.split(":", 1)[1].split("=")
                spec = (a, None, float(w)) if a in ("hops", "carry", "occ", "bmask", "sep_reset", "dedup", "gate_alpha", "occ_rec", "note_len") else ("gate", None, None) if a == "gate" else ("ptr_all", None, None) if a == "ptr_all" else ("key_id", None, float(w)) if a == "key_id" else (("key_mix", a[4:], float(w)) if a.startswith("kid_") else ("bind", a, float(w)))
            log.append(bind(r, rng, spec))
        elif op.startswith("grow:"):                 # forced form grow:<n neurons>
            log.append(add_neurons(r, rng, int(op.split(":")[1])))
        elif op == "grow":
            log.append(add_neurons(r, rng, grow))
        elif op.startswith("rule:"):                 # forced form rule:<key>=<int>
            k, v = op.split(":", 1)[1].split("=")
            r.rules[k] = int(v)
            log.append(f"rule {k} -> {v}")
        elif op.startswith("module:"):               # forced form module:bow=<shift>
            k, v = op.split(":", 1)[1].split("=")
            r.mod[k] = int(v)
            log.append(f"module {k} -> {v}")
        elif op == "module":
            log.append(module(r, rng))
        elif op == "subpop":
            log.append(subpop(r, rng))
        elif op == "rules":
            log.append(rules(r, rng))
        elif op == "edges":
            f = rng.choice([0.005, 0.02, 0.1])
            log += [add_edges(r, rng, f), del_edges(r, rng, f)]
        elif op == "leak":
            log.append(leak(r, rng, rng.uniform(0.02, 0.2)))
        elif op == "reroute":
            log.append(reroute(r, rng, rng.randint(4, 64)))
        elif op == "del_neurons":
            cap = (int((~r.is_chain & ~r.is_in).sum()) - 256) // 256 * 256
            log.append(del_neurons(r, rng, min(cap, rng.choice([256, 512, 1024]))))
    while not fits(r.N, r.mod["H"] if r.mod else 0, r.D):          # make room within the hardware limit
        log.append(del_neurons(r, rng, 256))
    mutate.last_ops = [o.split(":")[0] for o in ops]
    if all(o.startswith(("readout", "bind")) for o in ops):
        return log                                           # network unchanged: keep its tuning (clean A/B)
    echo = r.tune(corpus_ids)
    log.append(f"retuned: g_rec={r.g_rec} echo={echo}")
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--grow", type=int, default=0)
    a = ap.parse_args()
    from .evolve import corpus
    r = Reservoir.from_state(torch.load(a.src))
    print("\n".join(mutate(r, random.Random(a.seed), corpus()[0], a.grow)))
    torch.save(r.state_dict(), a.dst)
