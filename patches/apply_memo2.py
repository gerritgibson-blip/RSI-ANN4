"""Patch: per-step note write limit grows with the note_len gene (MEMO2_PATCH). Idempotent.
Finding (2026-11-27, CPU replay of the training labels): a 3-goal todo needs code, key, search words, file, site
and field. Each step may write at most 3 words (2 + carry2), and once the task text leaves view the dropped words
can never be written. So note_len 6 alone changed nothing (gen 798: todo 0.45, z -1.1). Now the cap is 2 + carry2
for note_len <= 4 (old behaviour, unchanged heads) and note_len - 1 otherwise; the ncarry head has cap + 1 classes.
Usage: python apply_memo2.py <dir containing trn/>"""
import sys, os
root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
A, Ev = (os.path.join(root, "trn", f) for f in ("agent.py", "evolve.py"))


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


a = open(A).read()
if "MEMO2_PATCH" not in a:
    a = sub(a, '''def carry_targets(env, ws_now, maxsteps=4, args_too=False):''', '''def carry_cap(cfg):
    """MEMO2_PATCH: most words one step may write into the note."""
    c2 = int((cfg or {}).get("carry") or 0) >= 2
    nl = int((cfg or {}).get("note_len") or 4)
    return 2 + c2 if nl <= 4 else nl - 1


def carry_targets(env, ws_now, maxsteps=4, args_too=False):''')
    a = sub(a, '''        self.carry2 = int((attn or {}).get("carry") or 0) >= 2           # STAGE5_PATCH: note may carry arguments''',
            '''        self.carry2 = int((attn or {}).get("carry") or 0) >= 2           # STAGE5_PATCH: note may carry arguments
        self.ccap = carry_cap(attn)                                       # MEMO2_PATCH''')
    a = sub(a, '''                         if u not in s.note][:2 + self.carry2]''', '''                         if u not in s.note][:self.ccap]   # MEMO2_PATCH''')
    a = sub(a, '''Fn.one_hot(torch.tensor([min(2 + self.carry2, len(streams[b].ctarget)) for b in drows], device=dev), 3 + self.carry2)''',
            '''Fn.one_hot(torch.tensor([min(self.ccap, len(streams[b].ctarget)) for b in drows], device=dev), self.ccap + 1)''')
    open(A, "w").write(a)

e = open(Ev).read()
if "carry_cap(cfg) + 1) if cfg and cfg.get(\"carry\") else 0))}" not in e:
    e = sub(e, '''([("ncarry", 3 + (int(cfg.get("carry")) >= 2))] if cfg and cfg.get("carry") else [])   # STAGE5_PATCH''',
            '''([("ncarry", __import__("trn.agent", fromlist=["carry_cap"]).carry_cap(cfg) + 1)] if cfg and cfg.get("carry") else [])   # STAGE5_PATCH MEMO2_PATCH''')
    e = sub(e, '''+ ((3 + (int(cfg.get("carry")) >= 2)) if cfg and cfg.get("carry") else 0))}''',
            '''+ ((__import__("trn.agent", fromlist=["carry_cap"]).carry_cap(cfg) + 1) if cfg and cfg.get("carry") else 0))}   # MEMO2_PATCH''')
    e = sub(e, '''            "hops": res.rules.get("hops", 1), "carry": res.rules.get("carry", 0)}     # REASON_PATCH''',
            '''            "hops": res.rules.get("hops", 1), "carry": res.rules.get("carry", 0), "note_len": res.rules.get("note_len", 4)}     # REASON_PATCH MEMO2_PATCH''')
    open(Ev, "w").write(e)
print("memo2 patch applied to", root)
