"""Patch on top of apply_alloc3: a family's training share may fall at most 15% per epoch (was 40%). Idempotent.
A/B (2026-11-27, live champion gen 698, seed 777, same gate tasks): epoch-70 shares 0.9462; shares with floor
0.85 * previous 0.9634 (z +7.8; research 0.76 -> 0.99 when its share went 3.8% -> 5.2%); 1.5x episodes 0.9645
(z +8.2) at +46% training time. Families sit at a data threshold: small share cuts cause collapses.
Usage: python apply_alloc4.py <dir containing trn/>"""
import sys, os
root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
Ev = os.path.join(root, "trn", "evolve.py")
e = open(Ev).read()
old = '''sh = {f: max(tgt[f], 0.6 * prev.get(f, 0.0)) for f in fams_}'''
new = '''sh = {f: max(tgt[f], 0.85 * prev.get(f, 0.0)) for f in fams_}   # ALLOC4_PATCH: floor 0.85'''
if "ALLOC4_PATCH" not in e:
    assert e.count(old) == 1
    open(Ev, "w").write(e.replace(old, new))
print("alloc4 patch applied to", root)
