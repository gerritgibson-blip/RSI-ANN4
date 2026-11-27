"""Patch: gene rules["occ"] -- occurrence linking for the answer pointer. Idempotent.

Diagnostic (champion gen 530, 2026-11-26): instr2 failures always copy the wrong OPTION ('answer a , b , or c'):
the pointer is trained on the last occurrence (the option list), where every name has the same context, so the
decision must be recalled from far back. 'v x , not y and not z' scored 26% (below chance). The negation /
correction cue is LOCAL at the instruction occurrence ('not y', 'i meant x', 'go back to x').
occ >= 1: pointer targets = every occurrence of a single-word answer (as ptr_all) AND at inference a word's unary
          score is the max over all occurrences of the same word (copying any occurrence gives the same word);
occ == 2: the attention score is occurrence-pooled the same way.
Usage: python apply_occ.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
A, Mu = (os.path.join(root, "trn", f) for f in ("agent.py", "mutate.py"))


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


a = open(A).read()
if "OCC_PATCH" not in a:
    a = sub(a, '''        s.labels = pointer_labels(s.obs_words, s.oracle[1], bool(self.res.rules.get("ptr_all"))) if s.oracle else None''',
            '''        s.labels = pointer_labels(s.obs_words, s.oracle[1], bool(self.res.rules.get("ptr_all") or self.res.rules.get("occ"))) if s.oracle else None''')
    a = sub(a, '''    def _attn_scores(self, k, q, s, ai):''', '''    @staticmethod
    def _occ_pool(v, ws):
        """OCC_PATCH: every position takes the max score over all occurrences of its word."""
        best = {}
        for i, x in enumerate(v):
            w = ws[i] if i < len(ws) else None
            if w not in best or x > best[w]:
                best[w] = x
        return np.array([best[ws[i] if i < len(ws) else None] for i in range(len(v))])

    def _attn_scores(self, k, q, s, ai):''')
    a = sub(a, '''            u = np.array([float(s.sel[w][ai]) if w in s.sel else -1e9 for w in range(len(a))])
            ok = u > -1e8''', '''            u = np.array([float(s.sel[w][ai]) if w in s.sel else -1e9 for w in range(len(a))])
            occ = int(self.res.rules.get("occ") or 0)
            if occ and ACTIONS[ai] == "answer":
                u = self._occ_pool(u, s.obs_words)
                if occ >= 2:
                    a = self._occ_pool(np.asarray(a), s.obs_words)
            ok = u > -1e8''')
    open(A, "w").write(a)

m = open(Mu).read()
if "OCC_PATCH" not in m:
    m = sub(m, '''"ptr_all", "gate", "hops", "carry"])''', '''"ptr_all", "gate", "hops", "carry", "occ"])   # OCC_PATCH''')
    m = sub(m, '''    if k in ("hops", "carry"):                                # REASON_PATCH''', '''    if k == "occ":
        cur = int(r.rules.get("occ") or 0)
        r.rules["occ"] = int(w) if spec else {0: 1, 1: 2, 2: 0}[cur]
        return f"readout occurrence linking -> {r.rules['occ']}"
    if k in ("hops", "carry"):                                # REASON_PATCH''')
    m = sub(m, '''spec = (a, None, float(w)) if a in ("hops", "carry") else''', '''spec = (a, None, float(w)) if a in ("hops", "carry", "occ") else''')
    open(Mu, "w").write(m)
print("occ patch applied to", root)
