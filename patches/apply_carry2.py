"""Patch on top of apply_reason: persistent note + learned note-write count (gene rules["carry"]=1). Idempotent.

Diagnostic (champion + carry, 2026-11-26): step-1 notes were right in 100% of chain episodes, but the note had
to be re-copied at every step (only ~50% succeeded) and a fixed threshold wrote spurious notes on 64% of web
decisions. Now: the note persists for the episode (max 4 words, oldest dropped), carry targets exclude words
already noted, and a 3-way decision head 'ncarry' (fitted in the decision Gram, like nargs) decides how many
of the carry pointer's top words to write (0, 1, 2).
Usage: python apply_carry2.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
A, Ev = os.path.join(root, "trn", "agent.py"), os.path.join(root, "trn", "evolve.py")


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


a = open(A).read()
if "CARRY2_PATCH" not in a:
    a = sub(a, '"carry", "ctarget", "clab", "dlab", "hop")', '"carry", "ctarget", "clab", "dlab", "hop", "note", "ncar")   # CARRY2_PATCH')
    a = sub(a, "        self.carry, self.ctarget, self.clab, self.dlab, self.hop = [], [], None, 0, 1\n",
            "        self.carry, self.ctarget, self.clab, self.dlab, self.hop, self.note, self.ncar = [], [], None, 0, 1, [], 0\n")
    a = sub(a, '''        s.carry, s.hop = [], 1
        self._push_obs(s, s.env.first)''', '''        s.carry, s.hop, s.note, s.ncar = [], 1, [], 0
        self._push_obs(s, s.env.first)''')
    a = sub(a, '''        if self.carry and s.carry:                       # self-written note carried from the previous observation
            obs = "note : " + " ".join(s.carry) + " . " + obs
        s.carry = []''', '''        if self.carry:                                   # persistent self-written note (max 4 words)
            for w_ in s.carry:
                if w_ not in s.note:
                    s.note.append(w_)
            s.note = s.note[-4:]
            if s.note:
                obs = "note : " + " ".join(s.note) + " . " + obs
        s.carry = []''')
    a = sub(a, '''            s.ctarget = carry_targets(s.env, s.obs_words)
''', '''            s.ctarget = [u for u in carry_targets(s.env, s.obs_words) if u not in s.note][:2]
''')
    a = sub(a, '''                    if self.hops >= 2:                      # chain-depth head (0 = no binding, 1, 2)
                        Ys.append(Fn.one_hot(torch.tensor([min(2, streams[b].dlab) for b in drows], device=dev), 3).float())''',
            '''                    if self.hops >= 2:                      # chain-depth head (0 = no binding, 1, 2)
                        Ys.append(Fn.one_hot(torch.tensor([min(2, streams[b].dlab) for b in drows], device=dev), 3).float())
                    if self.carry:                          # how many words to write into the note (0, 1, 2)
                        Ys.append(Fn.one_hot(torch.tensor([min(2, len(streams[b].ctarget)) for b in drows], device=dev), 3).float())''')
    a = sub(a, '''                if pred is not None and "hop" in self.heads:''', '''                if pred is not None and "ncarry" in self.heads:
                    for r_, n_ in enumerate(self._head("ncarry", Fd).argmax(1).tolist()):
                        streams[drows[r_]].ncar = n_
                if pred is not None and "hop" in self.heads:''')
    a = sub(a, '''                            cs_ = sorted(((float(v[self.cch]), w) for w, v in s.sel.items() if len(v) > self.cch), reverse=True)
                            s.carry = []
                            for v_, w_ in cs_[:2]:
                                if v_ > CARRY_T and w_ < len(s.obs_words) and s.obs_words[w_] not in PUNCT and s.obs_words[w_] not in s.carry:
                                    s.carry.append(s.obs_words[w_])''', '''                            cs_ = sorted(((float(v[self.cch]), w) for w, v in s.sel.items() if len(v) > self.cch), reverse=True)
                            s.carry = []
                            for v_, w_ in cs_:
                                if len(s.carry) >= s.ncar:
                                    break
                                if w_ < len(s.obs_words) and s.obs_words[w_] not in PUNCT and s.obs_words[w_] not in s.carry \\
                                        and s.obs_words[w_] not in s.note:
                                    s.carry.append(s.obs_words[w_])''')
    open(A, "w").write(a)

e = open(Ev).read()
if "CARRY2_PATCH" not in e:
    e = sub(e, '''        nh = getattr(acc[name], "nhop", 0)
        if nh:                                               # chain-depth head sits after the query columns
            heads["hop"] = W[:, W.shape[1] - nh:].contiguous()
            W = W[:, : W.shape[1] - nh]
''', '''        for tname, tn in reversed(getattr(acc[name], "tail", [])):   # CARRY2_PATCH: extra heads after the query columns
            heads[tname] = W[:, W.shape[1] - tn:].contiguous()
            W = W[:, : W.shape[1] - tn]
''')
    e = sub(e, '''           "dec": Ridge(D, len(ACTIONS) + 4 + (key_dim(cfg) if cfg else 0) + (3 if hop_on(cfg) else 0))}
    if hop_on(cfg):
        acc["dec"].nhop = 3''', '''           "dec": Ridge(D, len(ACTIONS) + 4 + (key_dim(cfg) if cfg else 0) + (3 if hop_on(cfg) else 0) + (3 if cfg and cfg.get("carry") else 0))}
    acc["dec"].tail = ([("hop", 3)] if hop_on(cfg) else []) + ([("ncarry", 3)] if cfg and cfg.get("carry") else [])''')
    open(Ev, "w").write(e)
print("carry2 patch applied to", root)
