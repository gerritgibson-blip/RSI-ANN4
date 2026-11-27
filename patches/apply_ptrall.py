"""Patch: gene rules["ptr_all"] -- for single-word arguments, every occurrence of the answer word is a pointer
target (copying any occurrence yields the same word), so the readout can point where the decision is LOCAL
(e.g. 'install y instead' rather than the option list 'answer x , or y'). Idempotent.
Usage: python apply_ptrall.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
Ev, A, Mu = (os.path.join(root, "trn", f) for f in ("env.py", "agent.py", "mutate.py"))


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


e = open(Ev).read()
e = sub(e, '''def pointer_labels(obs_words, args):''', '''def pointer_labels(obs_words, args, all_occ=False):''')
e = sub(e, '''        i = cand[-1] if len(args) == 1 else cand[0]
        pos.append(i); last = i
    for i in pos:''', '''        i = cand[-1] if len(args) == 1 else cand[0]
        pos.append(i); last = i
    if all_occ and len(args) == 1:                 # PTRALL: every occurrence of a single answer word is a target
        pos = [i for i, w in enumerate(obs_words) if w == args[0]]
    for i in pos:''')
open(Ev, "w").write(e)

a = open(A).read()
a = sub(a, "        s.labels = pointer_labels(s.obs_words, s.oracle[1]) if s.oracle else None\n",
        "        s.labels = pointer_labels(s.obs_words, s.oracle[1], bool(self.res.rules.get(\"ptr_all\"))) if s.oracle else None\n")
open(A, "w").write(a)

m = open(Mu).read()
m = sub(m, '''        k = rng.choice(["bind", "bind", "key_id", "key_mix", "key_mix"])
        a = rng.choice(ATTN_ACTIONS) if k in ("bind", "key_mix") else None''',
        '''        k = rng.choice(["bind", "bind", "key_id", "key_mix", "key_mix", "ptr_all"])
        a = rng.choice(ATTN_ACTIONS) if k in ("bind", "key_mix") else None
    if k == "ptr_all":
        r.rules["ptr_all"] = 0 if r.rules.get("ptr_all") else 1
        return f"readout pointer targets all occurrences -> {r.rules['ptr_all']}"''')
m = sub(m, '''                spec = ("key_id", None, float(w)) if a == "key_id" else''',
        '''                spec = ("ptr_all", None, None) if a == "ptr_all" else ("key_id", None, float(w)) if a == "key_id" else''')
open(Mu, "w").write(m)
print("ptr_all patch applied to", root)
