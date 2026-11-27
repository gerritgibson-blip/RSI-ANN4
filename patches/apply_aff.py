"""Patch: gene rules["aff"] -- affirmed-mention pointer supervision for instruction families (AFF_PATCH). Idempotent.

Analysis (2026-11-27): the answer pointer reads a word's score at the end of its clause (causal). instr
'v x . no wait , v y .' -> y and instr2 '... no wait , v y . actually , go back to x .' -> x are identical up to
y's clause end, but y's label is positive in one family and negative in the other, so ridge averages them: the
two families compete (share A/B: instr +6 points = instr2 -5 points). Consistent labelling: a mention is a target
iff it is AFFIRMED in its own clause (not negated); the answer is then the LATEST affirmed mention, which the
recency-aware occurrence pooling (occ_rec) selects. Every instr/instr2 form satisfies this rule; the envs record
which mentions are affirmed. This changes only the training labels of the answer pointer on these two families.
Usage: python apply_aff.py <dir containing trn/>"""
import sys, os
root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
Ev, A, Mu = (os.path.join(root, "trn", f) for f in ("env.py", "agent.py", "mutate.py"))


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, (src.count(old), old)
    return src.replace(old, new)


e = open(Ev).read()
if "AFF_PATCH" not in e:
    e = sub(e, '''        if form == 0:
            text = f"do not {verb} {x} . {verb} {y} instead ."
        elif form == 1:
            text = f"{verb} {x} . no wait , {verb} {y} ."
        else:
            text = f"{verb} {y} but not {x} ."''', '''        if form == 0:
            text = f"do not {verb} {x} . {verb} {y} instead ."
            self.aff = [(y, 0)]                       # AFF_PATCH: (name, k-th occurrence) mentions that are affirmed
        elif form == 1:
            text = f"{verb} {x} . no wait , {verb} {y} ."
            self.aff = [(x, 0), (y, 0)]
        else:
            text = f"{verb} {y} but not {x} ."
            self.aff = [(y, 0)]''')
    e = sub(e, '''        if form == 0:
            text, ans = f"{v} {x} . no wait , {v} {y} . actually , go back to {x} .", x
        elif form == 1:
            text, ans = f"{v} {y} . sorry , i meant {x} .", x
        elif form == 2:
            text, ans = f"do not {v} {x} or {y} . {v} {z} .", z
        else:
            text, ans = f"{v} {x} , not {y} and not {z} .", x''', '''        if form == 0:
            text, ans = f"{v} {x} . no wait , {v} {y} . actually , go back to {x} .", x
            self.aff = [(x, 0), (y, 0), (x, 1)]       # AFF_PATCH
        elif form == 1:
            text, ans = f"{v} {y} . sorry , i meant {x} .", x
            self.aff = [(y, 0), (x, 0)]
        elif form == 2:
            text, ans = f"do not {v} {x} or {y} . {v} {z} .", z
            self.aff = [(z, 0)]
        else:
            text, ans = f"{v} {x} , not {y} and not {z} .", x
            self.aff = [(x, 0)]''')
    open(Ev, "w").write(e)

a = open(A).read()
if "AFF_PATCH" not in a:
    a = sub(a, '''        s.labels = pointer_labels(s.obs_words, s.oracle[1], (2 if (self.res.rules.get("occ") == 3 and s.oracle[0] == "answer") else bool(self.res.rules.get("ptr_all") or self.res.rules.get("occ")))) if s.oracle else None''',
            '''        s.labels = pointer_labels(s.obs_words, s.oracle[1], (2 if (self.res.rules.get("occ") == 3 and s.oracle[0] == "answer") else bool(self.res.rules.get("ptr_all") or self.res.rules.get("occ")))) if s.oracle else None
        if self.res.rules.get("aff") and s.oracle and s.oracle[0] == "answer" and getattr(s.env, "aff", None):   # AFF_PATCH
            pos_ = []
            for nm_, k_ in s.env.aff:
                occ_ = [i for i, w in enumerate(s.obs_words) if w == nm_]
                if len(occ_) > k_:
                    pos_.append(occ_[k_])
            if pos_:
                s.labels = aff_labels(s.obs_words, pos_)''')
    a = sub(a, '''def carry_targets(env, ws_now, maxsteps=4, args_too=False):''', '''def aff_labels(obs_words, pos):
    """AFF_PATCH: pointer targets at the given word positions (read from the end of each word's clause)."""
    labels = [0] * len(obs_words)
    for i in pos:
        j = i
        while j < len(obs_words) - 1 and obs_words[j] not in PUNCT and j - i < MAXPTR - 1:
            j += 1
        labels[j] |= 1 << (j - i)
    return labels


def carry_targets(env, ws_now, maxsteps=4, args_too=False):''')
    open(A, "w").write(a)

m = open(Mu).read()
if "AFF_PATCH" not in m:
    m = sub(m, '''"sep_reset", "dedup", "gate_alpha", "occ_rec") else''', '''"sep_reset", "dedup", "gate_alpha", "occ_rec", "aff") else''')
    m = sub(m, '''    if k == "occ_rec":                                        # OCCREC_PATCH''', '''    if k == "aff":                                            # AFF_PATCH
        r.rules["aff"] = int(w) if spec else (0 if r.rules.get("aff") else 1)
        return f"affirmed-mention answer labels -> {r.rules['aff']}"
    if k == "occ_rec":                                        # OCCREC_PATCH''')
    m = sub(m, '''"dedup", "occ_rec"])   # DEDUP_PATCH OCCREC_PATCH''', '''"dedup", "occ_rec", "aff"])   # DEDUP_PATCH OCCREC_PATCH AFF_PATCH''')
    open(Mu, "w").write(m)
print("aff patch applied to", root)
