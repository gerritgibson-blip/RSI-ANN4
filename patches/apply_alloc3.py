"""Patch: rate-limited training allocation (ALLOC3_PATCH) + g_rec == 0 fast path (GREC0_PATCH). Idempotent.

Diagnosis (epochs 64-67, 2026-11-26): the share of training data per family follows 12*(1-acc) and is
recomputed from scratch every epoch. research2 had 16% of the data in epoch 65 and reached 0.99; epoch 66
saw 0.99 and cut it to 7.5% -> every mutant AND the re-fitted champion fell to research2 ~0.55; epoch 67 then
restores 17.5%. A solved family still needs the data that solved it.
Now: a family's share may fall by at most 40% per epoch (share = max(target, 0.6 * previous), renormalised).

g_rec: the evolved gene g_fix = 0 (since ~gen 480) zeroes the recurrent drive, yet every tick still computed
the recurrent product (dense int8 N x N or CSR) and multiplied it by 0, and the 'edges' operator (which only
changes W) was a no-op: gens 670/671/675 had bit-identical results (0.9467, z -1.11) and wasted ~1/4 of
generations. Now the product is skipped when g_rec == 0 (exact) and 'edges' is not drawn while g_rec == 0.
Usage: python apply_alloc3.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
Ev, R, Mu = (os.path.join(root, "trn", f) for f in ("evolve.py", "reservoir.py", "mutate.py"))


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


e = open(Ev).read()
if "ALLOC3_PATCH" not in e:
    e = sub(e, '''TRAIN_ACC = None      # ALLOC_PATCH''', '''TRAIN_EMPH = None     # ALLOC3_PATCH: rate-limited per-family data share (set per epoch by main)
TRAIN_ACC = None      # ALLOC_PATCH''')
    e = sub(e, '''        emph = {f: max(1.0, emph[f] / 3) + 12.0 * (1.0 - TRAIN_ACC.get(f, 0.5)) for f in fams}
''', '''        emph = {f: max(1.0, emph[f] / 3) + 12.0 * (1.0 - TRAIN_ACC.get(f, 0.5)) for f in fams}
    if TRAIN_EMPH and set(TRAIN_EMPH) == set(fams):     # ALLOC3_PATCH
        emph = dict(TRAIN_EMPH)
''')
    e = sub(e, '''    global TRAIN_ACC
    TRAIN_ACC = st.get("train_acc")
''', '''    global TRAIN_ACC, TRAIN_EMPH
    TRAIN_ACC = st.get("train_acc")
    TRAIN_EMPH = st.get("train_emph")                 # ALLOC3_PATCH
''')
    e = sub(e, '''            TRAIN_ACC = st.get("train_acc")
            log("epoch", epoch=st["epoch"], train_acc=TRAIN_ACC)''', '''            TRAIN_ACC = st.get("train_acc")
            if TRAIN_ACC:                             # ALLOC3_PATCH: share may fall by at most 40% per epoch
                fams_ = active(st["stage"])
                tgt = {f: (max(1.0, (1.0 + FAMILIES[f].stage) * (1.5 if FAMILIES[f].stage == st["stage"] else 1.0) / 3)
                           + 12.0 * (1.0 - TRAIN_ACC.get(f, 0.5))) for f in fams_}
                tt = sum(tgt.values()); tgt = {f: v / tt for f, v in tgt.items()}
                prev = st.get("train_emph") or {}
                sh = {f: max(tgt[f], 0.6 * prev.get(f, 0.0)) for f in fams_}
                ss = sum(sh.values())
                st["train_emph"] = TRAIN_EMPH = {f: round(v / ss, 5) for f, v in sh.items()}
            log("epoch", epoch=st["epoch"], train_acc=TRAIN_ACC, train_share=TRAIN_EMPH)''')
    open(Ev, "w").write(e)

r = open(R).read()
if "GREC0_PATCH" not in r:
    r = sub(r, '''        if csr is not None:
            I = (csr @ st["s"].T.contiguous().float()).T.contiguous().to(torch.int32)''', '''        if self.g_rec == 0:                               # GREC0_PATCH: no recurrent drive -> skip the product (exact)
            I = 0
        elif csr is not None:
            I = (csr @ st["s"].T.contiguous().float()).T.contiguous().to(torch.int32)''')
    open(R, "w").write(r)

m = open(Mu).read()
if "GREC0_PATCH" not in m:
    m = sub(m, '''        if "del_neurons" in ops and int((~r.is_chain & ~r.is_in).sum()) < 512:''', '''        if "edges" in ops and r.g_rec == 0 and r.rules.get("g_fix") == 0:   # GREC0_PATCH: W is unused -> no-op
            ops.remove("edges")
        if "del_neurons" in ops and int((~r.is_chain & ~r.is_in).sum()) < 512:''')
    open(Mu, "w").write(m)
print("alloc3 + grec0 patch applied to", root)
