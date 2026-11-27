"""Patch: gene rules["bmask"] -- the bridge (focus) pointer may only select words that occur at least twice in the
observation. Idempotent.

Diagnostic (champion gen 530 + occ, 2026-11-26): chain 'river' questions succeed 51%, mayor 100%, population 88%.
At the answer decision the note always held the key ('note : river capital .') and the hop head said 1, but the
focus pointer chose 'capital' (adjacent in the note; it occurs nowhere else), so the identity query bound the
note's own clause and the river value got no binding bonus. Mayor / population answers survive on unary cues
(person names, 'million'); made-up river names depend on binding. A word occurring once cannot be a bridge.
Usage: python apply_bmask.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
A, Mu = (os.path.join(root, "trn", f) for f in ("agent.py", "mutate.py"))


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


a = open(A).read()
if "BMASK_PATCH" not in a:
    a = sub(a, '''        f = np.array([float(s.sel[w][fc]) if (w in s.sel and len(s.sel[w]) > fc) else -1e9 for w in range(n)])
        ok = f > -1e8''', '''        f = np.array([float(s.sel[w][fc]) if (w in s.sel and len(s.sel[w]) > fc) else -1e9 for w in range(n)])
        if self.res.rules.get("bmask"):                 # BMASK_PATCH: a bridge word must occur elsewhere too
            cnt_ = collections.Counter(ws)
            f = np.where(np.array([cnt_[w] >= 2 for w in ws]), f, -1e9)
        ok = f > -1e8''')
    open(A, "w").write(a)

m = open(Mu).read()
if "BMASK_PATCH" not in m:
    m = sub(m, '''"carry", "occ"])   # OCC_PATCH''', '''"carry", "occ", "bmask"])   # OCC_PATCH BMASK_PATCH''')
    m = sub(m, '''    if k == "occ":''', '''    if k == "bmask":
        r.rules["bmask"] = int(w) if spec else (0 if r.rules.get("bmask") else 1)
        return f"readout bridge needs a repeated word -> {r.rules['bmask']}"
    if k == "occ":''')
    m = sub(m, '''if a in ("hops", "carry", "occ") else''', '''if a in ("hops", "carry", "occ", "bmask") else''')
    open(Mu, "w").write(m)
print("bmask patch applied to", root)
