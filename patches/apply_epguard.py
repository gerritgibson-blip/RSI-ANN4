"""Patch: epoch training-sample guard (EPGUARD_PATCH). Idempotent.
Finding (2026-11-27, epoch 78, train_seed 617783): refitting on that epoch's training sample broke the answer
readout for instr/instr2 for EVERY genome -- record gen 740 refit: instr 0.995 -> 0.805, instr2 0.963 -> 0.862
on the same gate set, gate on or off; its stored heads still scored 0.995/0.963 on the epoch-78 gate set. So
rare training samples are degenerate. Guard: after the epoch champion refit, if its gate fitness is more than
EPGUARD_DROP below the record's final fitness, draw another train_seed (up to 2 retries, keep the best).
A degenerate epoch is not entered into acc_hist (it would inflate the allocation for 4 epochs); a manual
st["bad_epoch"] = True also skips the append once.
Usage: python apply_epguard.py <dir containing trn/>"""
import sys, os
root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
Ev = os.path.join(root, "trn", "evolve.py")


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


e = open(Ev).read()
if "EPGUARD_PATCH" not in e:
    e = sub(e, '''            if st.get("alloc"):                       # ALLOC2_PATCH: damped -- min over the last 4 epochs
                st.setdefault("acc_hist", []).append(dict(st["best"]["families"]))''',
            '''            if st.get("alloc") and not st.pop("bad_epoch", False):   # ALLOC2_PATCH / EPGUARD_PATCH
                st.setdefault("acc_hist", []).append(dict(st["best"]["families"]))''')
    e = sub(e, '''            log("epoch", epoch=st["epoch"], train_acc=TRAIN_ACC, train_share=TRAIN_EMPH)
            rebaseline(None)''', '''            log("epoch", epoch=st["epoch"], train_acc=TRAIN_ACC, train_share=TRAIN_EMPH)
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
                rebaseline(None)''')
    e = sub(e, '''TRAIN_EMPH = None''', '''EPGUARD_DROP = float(os.environ.get("TRN_EPGUARD_DROP", 0.01))   # EPGUARD_PATCH
TRAIN_EMPH = None''')
    open(Ev, "w").write(e)
print("epguard patch applied to", root)
