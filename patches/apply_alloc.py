"""Patch: error-driven allocation of readout training episodes across task families. Idempotent.

Component analysis on the champion (gen 476) showed the unary pointer never learned instr (43-53% alone),
while a stage-0 lab with 1/3 of the data on instr learned it to 95%: under the fixed curriculum emphasis
(stage-3 families x6) instr got 3% of training episodes although stage-3 families are at 100%.
Weights: w_f = max(1, emph_f / 3) + 12 * (1 - acc_f), acc_f = the champion's gate accuracy; recomputed at
every epoch start (fixed within an epoch, so champion and mutants still train on identical data).
Usage: python apply_alloc.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
Ev = os.path.join(root, "trn", "evolve.py")


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


e = open(Ev).read()
if "ALLOC_PATCH" not in e:
    e = sub(e, "_CORPUS = None\n", "_CORPUS = None\nTRAIN_ACC = None      # ALLOC_PATCH: champion gate accuracy per family (set per epoch by main)\n")
    e = sub(e, '''    emph = {f: (1.0 + FAMILIES[f].stage) * (1.5 if FAMILIES[f].stage == stage else 1.0) for f in fams}   # agentic + newest skills weigh most
''', '''    emph = {f: (1.0 + FAMILIES[f].stage) * (1.5 if FAMILIES[f].stage == stage else 1.0) for f in fams}   # agentic + newest skills weigh most
    if TRAIN_ACC:                                     # error-driven allocation: data goes where the failures are
        emph = {f: max(1.0, emph[f] / 3) + 12.0 * (1.0 - TRAIN_ACC.get(f, 0.5)) for f in fams}
''')
    e = sub(e, '''        if st["epoch_gens"] >= EPOCH:
            st["epoch"] += 1; st["epoch_gens"] = 0; st["train_seed"] = 101 + 7919 * st["epoch"]
            log("epoch", epoch=st["epoch"])
            rebaseline(None)
''', '''        if st["epoch_gens"] >= EPOCH:
            st["epoch"] += 1; st["epoch_gens"] = 0; st["train_seed"] = 101 + 7919 * st["epoch"]
            if st.get("alloc"):
                st["train_acc"] = dict(st["best"]["families"])
            TRAIN_ACC = st.get("train_acc")
            log("epoch", epoch=st["epoch"], train_acc=TRAIN_ACC)
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
''')
    e = sub(e, '''    if st.get("readout_version", 1) != 3:''', '''    global TRAIN_ACC
    TRAIN_ACC = st.get("train_acc")
    if st.get("readout_version", 1) != 3:''')
    open(Ev, "w").write(e)
print("alloc patch applied to", root)
