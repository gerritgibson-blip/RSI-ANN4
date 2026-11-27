"""Patch: stratified, higher-resolution evaluation (EVAL2_PATCH). Idempotent.

Stall diagnosis (gens 557-634, 2026-11-26): fitness ~0.958, the remaining ~4% error sits in 6 families, and the
remaining improvements are ~0.003-0.005 -- below the resolution of 150 tasks / family (the champion's own instr
gate score moved 0.79 <-> 0.97 between gate sets; gen 614 was accepted on gate noise with final z -2.05; gen 627
is likely better than the record, final 0.961 vs 0.958, but z 1.1 < 1.5). Families at 100% add ~no variance,
so precision is bought where the errors are: each family whose record accuracy is < 0.99 gets EVAL_MULT x the
tasks (fitness still weighs every family equally). Evaluation is cheap (~0.1-0.2 min / set vs ~6 min training).
Migration (once): fix st["eval_mult"], re-measure the record network on the new final set, re-baseline the
champion on gate/dev, then judge the champion against the record on the final set (ADVANCE if z >= 1.5).
Usage: python apply_eval2.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
Ev = os.path.join(root, "trn", "evolve.py")


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


e = open(Ev).read()
if "EVAL2_PATCH" not in e:
    e = sub(e, "_CORPUS = None\n", "_CORPUS = None\nEVAL_MULT = {}        # EVAL2_PATCH: family -> evaluation-task multiplier (fixed per stage, st['eval_mult'])\nEVAL_MULT_K = 4\n")
    e = sub(e, '''    fams = active(stage)
    run = Runner(res, heads, corpus()[0], attn=attn_cfg(res))
    results, _ = run.run(specs_for(fams, n, base, "test"), mode="agent", B=B)
    results.sort(key=lambda r: (r["family"], r["seed"]))
    lm, lm_items = run.lm_eval(corpus()[1], seed={"gate": 1000 + epoch, "dev": 22, "final": 33}[which])
    fam = {f: sum(r["success"] for r in results if r["family"] == f) / n for f in fams}
    wl = w_lm(stage)
    fitness = (1 - wl) * sum(fam.values()) / len(fams) + wl * lm
    items = [(float(r["success"]), (1 - wl) / (len(fams) * n)) for r in results] + [(x, wl / len(lm_items)) for x in lm_items]''',
            '''    fams = active(stage)
    run = Runner(res, heads, corpus()[0], attn=attn_cfg(res))
    nf = {f: n * int(EVAL_MULT.get(f, 1)) for f in fams}
    specs = [(f, base + 1000 * i + k, "test") for k, f in enumerate(fams) for i in range(nf[f])]
    results, _ = run.run(specs, mode="agent", B=B)
    results.sort(key=lambda r: (r["family"], r["seed"]))
    lm, lm_items = run.lm_eval(corpus()[1], seed={"gate": 1000 + epoch, "dev": 22, "final": 33}[which])
    fam = {f: sum(r["success"] for r in results if r["family"] == f) / nf[f] for f in fams}
    wl = w_lm(stage)
    fitness = (1 - wl) * sum(fam.values()) / len(fams) + wl * lm
    items = [(float(r["success"]), (1 - wl) / (len(fams) * nf[r["family"]])) for r in results] + [(x, wl / len(lm_items)) for x in lm_items]''')
    e = sub(e, '''    if "best_dev_items" not in st:
        st["best_dev_items"] = evaluate(res, heads, st["stage"], "dev")["items"]''', '''    global EVAL_MULT
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
        st["best_dev_items"] = evaluate(res, heads, st["stage"], "dev")["items"]''')
    open(Ev, "w").write(e)
print("eval2 patch applied to", root)
