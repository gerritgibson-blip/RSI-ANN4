"""Patch: oracle working-memory hints + evolvable note length (MEMO_PATCH). Idempotent.

Probe (2026-11-27, stage-6 champion, 300 test episodes per family):
  cond  match 0.19 / nomatch 0.77 -- at the branching step the task's language word is out of view and was never
        written into the note: carry targets are only future ARGUMENTS and answer-clause bridges, never words that
        decide WHICH action comes next. The branch was undecidable. Fix: an env may define memo() -> words the
        oracle's future decisions depend on; carry_targets carries them (when they are about to leave view) like
        arguments. Cond.memo = [the task's language word].
  todo  2-goal plans 1.0, 3-goal plans 0.0: always fails at 'open the file F on S' (3-4 steps after F, S were seen);
        the 4-word note cannot hold code, key, file, site, field. Fix: gene rules["note_len"] (default 4 = old
        behaviour; mutation steps along 4, 6, 8; forced form bind:note_len=<n>).
Usage: python apply_memo.py <dir containing trn/>"""
import sys, os
root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
A, Ev, Mu = (os.path.join(root, "trn", f) for f in ("agent.py", "env.py", "mutate.py"))


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


a = open(A).read()
if "MEMO_PATCH" not in a:
    a = sub(a, '''    now, tg, k = set(ws_now), [], 0
    atg = []                                          # STAGE5_PATCH: argument targets (args first, then bridge words)
    nxt = set(words(obs))                             # carry an argument only when it is about to leave view''',
            '''    now, tg, k = set(ws_now), [], 0
    atg = []                                          # STAGE5_PATCH: argument targets (args first, then bridge words)
    nxt = set(words(obs))                             # carry an argument only when it is about to leave view
    if args_too and hasattr(env, "memo"):             # MEMO_PATCH: words a future DECISION depends on
        atg += [u for u in env.memo() if u in now and u not in atg]   # carried even if it reappears: it is the comparand''')
    a = sub(a, '''            s.note = s.note[-4:]''', '''            s.note = s.note[-int(self.res.rules.get("note_len") or 4):]   # MEMO_PATCH: evolvable note length''')
    open(A, "w").write(a)

e = open(Ev).read()
if "MEMO_PATCH" not in e:
    e = sub(e, '''    def goal_reached(self):
        return self.saved

    def oracle(self):
        if not self.looked:
            return "lookup", [self.c]''', '''    def goal_reached(self):
        return self.saved

    def memo(self):                                    # MEMO_PATCH: the branch depends on the task's language word
        return [] if self.looked else [self.lang]

    def oracle(self):
        if not self.looked:
            return "lookup", [self.c]''')
    open(Ev, "w").write(e)

m = open(Mu).read()
if "MEMO_PATCH" not in m:
    m = sub(m, '''"sep_reset", "dedup", "gate_alpha", "occ_rec") else''', '''"sep_reset", "dedup", "gate_alpha", "occ_rec", "note_len") else''')
    m = sub(m, '''"sep_reset", "dedup", "occ_rec"])   # DEDUP_PATCH OCCREC_PATCH''', '''"sep_reset", "dedup", "occ_rec", "note_len"])   # DEDUP_PATCH OCCREC_PATCH MEMO_PATCH''')
    m = sub(m, '''    if k == "occ_rec":                                        # OCCREC_PATCH''', '''    if k == "note_len":                                       # MEMO_PATCH
        L = [4, 6, 8]
        cur = int(r.rules.get("note_len") or 4)
        i = L.index(cur) if cur in L else 0
        r.rules["note_len"] = int(w) if spec else L[max(0, min(len(L) - 1, i + rng.choice([-1, 1])))]
        return f"note length -> {r.rules['note_len']}"
    if k == "occ_rec":                                        # OCCREC_PATCH''')
    open(Mu, "w").write(m)
print("memo patch applied to", root)
