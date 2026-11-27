"""Patch: module bag-of-words decay shift may evolve up to 8 (was capped at 5). Idempotent.
Probe (2026-11-27): toolpick2 fails where the phrase precedes the frame ('{x} , thanks .': ~16 tokens between the
phrase and the decision; recency wins, 'a place to keep {n}' -> lookup, test 23%). Evolution reached bow 5 = cap.
A/B on the champion (seed 777): bow5 0.9667, bow6 0.9683 (z 1.8), bow7 0.9692 (z 3.1; toolpick2 0.897 -> 0.925).
Usage: python apply_bow3.py <dir containing trn/>"""
import sys, os
root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
Mu = os.path.join(root, "trn", "mutate.py")
m = open(Mu).read()
old = '''m["bow"] = rng.choice([x for x in (0, 2, 3, 4, 5) if x != m.get("bow", 0)])'''
new = '''m["bow"] = rng.choice([x for x in (0, 2, 3, 4, 5, 6, 7, 8) if x != m.get("bow", 0)])   # BOW3_PATCH'''
if "BOW3_PATCH" not in m:
    assert m.count(old) == 1
    open(Mu, "w").write(m.replace(old, new))
print("bow3 patch applied to", root)
