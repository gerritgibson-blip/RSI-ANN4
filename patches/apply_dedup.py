"""Patch: gene rules["dedup"] -- argument copying never picks the same word twice (except calc, whose two
numbers may be equal) and never copies punctuation. Idempotent.

Trace (stage 5, carry=2, 2026-11-26): research fell 100 -> 15% because two-argument copies picked one name twice
('recall dunzo dunzo'): with the note, a name occurs in the note AND the text and both occurrences score high.
research2 also copied punctuation ('remember : talri').
Usage: python apply_dedup.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
A, Mu = (os.path.join(root, "trn", f) for f in ("agent.py", "mutate.py"))


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


a = open(A).read()
if "DEDUP_PATCH" not in a:
    a = sub(a, '''                            top = sorted(range(len(sc)), key=lambda w: sc[w], reverse=True)[: nargs[r]]
                        else:
                            top = sorted(s.sel, key=lambda w: float(s.sel[w][ai]), reverse=True)[: nargs[r]] if a != "done" else []''',
            '''                            top = sorted(range(len(sc)), key=lambda w: sc[w], reverse=True)
                        else:
                            top = sorted(s.sel, key=lambda w: float(s.sel[w][ai]), reverse=True) if a != "done" else []
                        if self.res.rules.get("dedup"):                     # DEDUP_PATCH: distinct, non-punctuation words
                            seen_, keep_ = set(), []
                            for i in top:
                                w_ = s.obs_words[i] if i < len(s.obs_words) else None
                                if w_ is None or w_ in PUNCT or (w_ in seen_ and a != "calc"):
                                    continue
                                seen_.add(w_); keep_.append(i)
                            top = keep_
                        top = top[: nargs[r]]''')
    open(A, "w").write(a)

m = open(Mu).read()
if "DEDUP_PATCH" not in m:
    m = sub(m, '''"bmask", "sep_reset"])''', '''"bmask", "sep_reset", "dedup"])   # DEDUP_PATCH''')
    m = sub(m, '''    if k == "sep_reset":''', '''    if k == "dedup":
        r.rules["dedup"] = int(w) if spec else (0 if r.rules.get("dedup") else 1)
        return f"argument copy distinct words -> {r.rules['dedup']}"
    if k == "sep_reset":''')
    m = sub(m, '''"bmask", "sep_reset") else''', '''"bmask", "sep_reset", "dedup") else''')
    open(Mu, "w").write(m)
print("dedup patch applied to", root)
