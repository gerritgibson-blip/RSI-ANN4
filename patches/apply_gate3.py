"""Patch (GATE3_PATCH): (1) readout-only mutations no longer reuse the champion's heads while the contextual gate
is on -- the gate head was fitted for the old attention mix (gen 755 'attention calc x0.5': z -10.8, previously a
harmless z 0.0); (2) the gate's fresh-episode budget is a gene rules["gate_ep"] (divisor of EP_ROUND0; default 8).
Idempotent. Usage: python apply_gate3.py <dir containing trn/>"""
import sys, os
root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
Ev = os.path.join(root, "trn", "evolve.py")


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


e = open(Ev).read()
if "GATE3_PATCH" not in e:
    e = sub(e, '''        if set(M.mutate.last_ops) == {"readout"} and "q" in heads:''',
            '''        if set(M.mutate.last_ops) == {"readout"} and "q" in heads and not mut.rules.get("gate"):   # GATE3_PATCH''')
    e = sub(e, '''max(4, EP_ROUND0 // 8), seed + 2), mode="gate"''',
            '''max(4, EP_ROUND0 // int(res.rules.get("gate_ep") or 8)), seed + 2), mode="gate"''')
    open(Ev, "w").write(e)
print("gate3 patch applied to", root)
