"""Patch: load the evaluation multipliers BEFORE any start-up re-baseline (EVALFIX_PATCH). Idempotent.
Bug (2026-11-27): main() set EVAL_MULT (x4 tasks for 6 families) only after the start-up rebaselines
(best is None / readout_version). The re-baselines at 17:48 and 19:02 therefore stored best_items / record_items
with 3664 items while every later evaluation had 6364. paired_z zips the lists, so every mutant comparison since
17:48 paired unrelated episodes. Gen 800 (note_len 6) logged delta -0.0024; correctly paired it is +0.0173, z +6.28.
Usage: python apply_evalfix.py <dir containing trn/>"""
import sys, os
root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
Ev = os.path.join(root, "trn", "evolve.py")
e = open(Ev).read()
if "EVALFIX_PATCH" not in e:
    old1 = '''    global TRAIN_ACC, TRAIN_EMPH
    TRAIN_ACC = st.get("train_acc")'''
    new1 = '''    global TRAIN_ACC, TRAIN_EMPH, EVAL_MULT
    EVAL_MULT = st.get("eval_mult") or {}             # EVALFIX_PATCH: before any rebaseline below
    TRAIN_ACC = st.get("train_acc")'''
    old2 = '''    global EVAL_MULT
    EVAL_MULT = st.get("eval_mult") or {}
    if st.get("eval_v") != 2'''
    new2 = '''    EVAL_MULT = st.get("eval_mult") or {}
    if st.get("eval_v") != 2'''
    assert e.count(old1) == 1 and e.count(old2) == 1
    e = e.replace(old1, new1).replace(old2, new2)
    open(Ev, "w").write(e)
print("evalfix patch applied to", root)
