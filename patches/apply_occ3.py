"""Patch on top of apply_occ: occ == 3 -- for a repeated single-word answer, pointer targets are all occurrences
EXCEPT the last one (the option list), which therefore becomes a negative for every name. Idempotent.

Probe (champion gen 550, occ=1): on 'v x , not y and not z' (45%) and 'v x but not y' (70%) the unary pointer at
the instruction occurrence is often right (0.44 vs 0.04/0.22), but option-list occurrences -- trained positive under
occ=1 although their context is uninformative -- score 0.2-0.39 for every name and win the max pooling.
Also: attention pooling is now occ == 2 only. Exclusion applies to the answer action only (v1 also changed the
open-action targets, which are not occurrence-pooled: web 99 -> 67%, webdeep 100 -> 53%).
Usage: python apply_occ3.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
Ev, A, Mu = (os.path.join(root, "trn", f) for f in ("env.py", "agent.py", "mutate.py"))


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


e = open(Ev).read()
e = sub(e, '''        pos = [i for i, w in enumerate(obs_words) if w == args[0]]
''', '''        pos = [i for i, w in enumerate(obs_words) if w == args[0]]
        if all_occ == 2 and len(pos) > 1:          # OCC3_PATCH: the last (option-list) occurrence is not a target
            pos = pos[:-1]
''')
open(Ev, "w").write(e)

a = open(A).read()
a = sub(a, '''bool(self.res.rules.get("ptr_all") or self.res.rules.get("occ"))''',
        '''(2 if (self.res.rules.get("occ") == 3 and s.oracle[0] == "answer") else bool(self.res.rules.get("ptr_all") or self.res.rules.get("occ")))''')
a = sub(a, '''                if occ >= 2:
                    a = self._occ_pool(np.asarray(a), s.obs_words)''', '''                if occ == 2:                                  # OCC3_PATCH
                    a = self._occ_pool(np.asarray(a), s.obs_words)''')
open(A, "w").write(a)

m = open(Mu).read()
m = sub(m, '''{0: 1, 1: 2, 2: 0}[cur]''', '''{0: 1, 1: 3, 3: 2, 2: 0}[cur]''')
open(Mu, "w").write(m)
print("occ3 patch applied to", root)
