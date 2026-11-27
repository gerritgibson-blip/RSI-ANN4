"""Patch: gene rules["sep_reset"] -- clear the word-match fast-weight memory M and the bag-of-words trace at each
episode start (SEP token). Idempotent.

Finding (2026-11-26): episodes start with [SEP, PAD, PAD] only; M decays ~2^-5 per word and is never cleared, so
common words ('not', 'which', 'answer', option words) retrieve associations stored during the PREVIOUS task.
Clearing at SEP without any retraining: gate fitness 0.9554 -> 0.9584, toolpick2 0.892 -> 0.927, instr 0.967 -> 0.983.
Usage: python apply_sepreset.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
R, Mu = (os.path.join(root, "trn", f) for f in ("reservoir.py", "mutate.py"))


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


r = open(R).read()
if "SEPRESET_PATCH" not in r:
    r = sub(r, '''        m = self.mod
        sep = (tokens < 3) | (tokens == SPACE)
        h = st["h"]''', '''        m = self.mod
        if self.rules.get("sep_reset"):                  # SEPRESET_PATCH: a new episode starts with an empty memory
            keep = (tokens != SEP).float()
            st["M"].mul_(keep[:, None, None])
            if "bw" in st:
                st["bw"].mul_(keep[:, None])
        sep = (tokens < 3) | (tokens == SPACE)
        h = st["h"]''')
    open(R, "w").write(r)

m = open(Mu).read()
if "SEPRESET_PATCH" not in m:
    m = sub(m, '''"carry", "occ", "bmask"])   # OCC_PATCH BMASK_PATCH''', '''"carry", "occ", "bmask", "sep_reset"])   # OCC_PATCH BMASK_PATCH SEPRESET_PATCH''')
    m = sub(m, '''    if k == "bmask":''', '''    if k == "sep_reset":
        r.rules["sep_reset"] = int(w) if spec else (0 if r.rules.get("sep_reset") else 1)
        return f"episode-start memory reset -> {r.rules['sep_reset']}"
    if k == "bmask":''')
    m = sub(m, '''if a in ("hops", "carry", "occ", "bmask") else''', '''if a in ("hops", "carry", "occ", "bmask", "sep_reset") else''')
    open(Mu, "w").write(m)
print("sep_reset patch applied to", root)
