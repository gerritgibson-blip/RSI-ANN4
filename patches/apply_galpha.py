"""Patch: gene rules["gate_alpha"] -- the gate v3 shrinkage strength toward the default mix evolves (GALPHA_PATCH).
A/B (2026-11-27, champion, 4 hard negatives): alpha 10 z +3.0, alpha 1 z +3.6, alpha 0.1 z +4.1 vs gate off; the
deployed default was 1.0 (env TRN_GATE_ALPHA). When the gate is on, half of the 'gate' mutations now move alpha
one step along (0.03, 0.1, 0.3, 1, 3) instead of switching the gate off. Forced form: bind:gate_alpha=<value>.
Idempotent. Usage: python apply_galpha.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
Ev, Mu = (os.path.join(root, "trn", f) for f in ("evolve.py", "mutate.py"))


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


e = open(Ev).read()
if "GALPHA_PATCH" not in e:
    e = sub(e, '''                heads["gate"] = gf.solve(GATE_ALPHA, w0=run.gate_prior(gf.ds))''',
            '''                heads["gate"] = gf.solve(float(res.rules.get("gate_alpha") or GATE_ALPHA), w0=run.gate_prior(gf.ds))   # GALPHA_PATCH''')
    open(Ev, "w").write(e)

m = open(Mu).read()
if "GALPHA_PATCH" not in m:
    m = sub(m, '''    if k == "gate":                                           # GATE_PATCH
        r.rules["gate"] = 0 if r.rules.get("gate") else 2      # GATE2_PATCH: v3 gate''', '''    if k == "gate_alpha" or (k == "gate" and not spec and r.rules.get("gate") == 2 and rng.random() < 0.5):   # GALPHA_PATCH
        L = [0.03, 0.1, 0.3, 1.0, 3.0]
        cur = float(r.rules.get("gate_alpha") or 1.0)
        i = min(range(len(L)), key=lambda j: abs(L[j] - cur))
        r.rules["gate_alpha"] = float(w) if (spec and k == "gate_alpha") else L[max(0, min(len(L) - 1, i + rng.choice([-1, 1])))]
        return f"readout gate shrinkage -> {r.rules['gate_alpha']}"
    if k == "gate":                                           # GATE_PATCH
        r.rules["gate"] = 0 if r.rules.get("gate") else 2      # GATE2_PATCH: v3 gate''')
    m = sub(m, '''if a in ("hops", "carry", "occ", "bmask", "sep_reset", "dedup") else''',
            '''if a in ("hops", "carry", "occ", "bmask", "sep_reset", "dedup", "gate_alpha") else''')
    open(Mu, "w").write(m)
print("galpha patch applied to", root)
