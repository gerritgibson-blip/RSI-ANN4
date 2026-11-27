"""Patch: damped error-driven allocation (ALLOC2_PATCH). Idempotent.

Diagnosis (epochs 57-64, 2026-11-26): allocation used the champion's accuracy from the previous epoch only, which
is a bang-bang feedback loop: instr 0.82 -> 0.97 -> 0.77 -> 0.99 -> 0.80 -> 0.97 -> 0.79 -> 0.97 (instr2 trades
against it). Every other epoch the champion is ~0.01 worse, so a mutant accepted in a 'bad' epoch cannot beat a
record measured in a 'good' one on the final set -- a handicap larger than any remaining gain.
Now: allocation accuracy = the MINIMUM over the last 4 epochs (a family that recently failed keeps its data).
Usage: python apply_alloc2.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
Ev = os.path.join(root, "trn", "evolve.py")


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


e = open(Ev).read()
if "ALLOC2_PATCH" not in e:
    e = sub(e, '''            if st.get("alloc"):
                st["train_acc"] = dict(st["best"]["families"])''', '''            if st.get("alloc"):                       # ALLOC2_PATCH: damped -- min over the last 4 epochs
                st.setdefault("acc_hist", []).append(dict(st["best"]["families"]))
                st["acc_hist"] = st["acc_hist"][-4:]
                st["train_acc"] = {f: min(h.get(f, 1.0) for h in st["acc_hist"]) for f in st["best"]["families"]}''')
    open(Ev, "w").write(e)
print("alloc2 patch applied to", root)
