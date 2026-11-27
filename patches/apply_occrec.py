"""Patch: gene rules["occ_rec"] = beta -- recency-aware occurrence pooling of the unary answer score (OCCREC_PATCH).
Idempotent.

Analysis (2026-11-27): instr 'v x . no wait , v y .' -> y, instr2 '... no wait , v y . actually , go back to x .' -> x.
The unary pointer is causal: at y's position the two texts are identical, yet the labels are opposite, so ridge
averages them and both families lose. Occurrence pooling takes max over a word's mentions and discards WHERE they
were. In every instruction form the decisive mention is the latest strong one ('no wait , v y', 'go back to x',
'i meant x', '... . v z'). Now: pooled(w) = max_i (z_i + beta * i / n) over the word's occurrences, z = z-scored
unary score (answer action, occ pooling on). beta = 0 reproduces the previous behaviour.
Usage: python apply_occrec.py <dir containing trn/>"""
import sys, os
root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
A, Mu = (os.path.join(root, "trn", f) for f in ("agent.py", "mutate.py"))


def sub(src, old, new, cnt=1):
    if new in src:
        return src
    assert src.count(old) == cnt, (src.count(old), old)
    return src.replace(old, new)


a = open(A).read()
if "OCCREC_PATCH" not in a:
    # recency-aware pooling helper
    a = sub(a, '''    @staticmethod
    def _occ_pool(v, ws):''', '''    def _occ_pool_rec(self, u, ws):
        """OCCREC_PATCH: z-score the candidate scores, add beta * position / n, then pool over occurrences."""
        beta = float(self.res.rules.get("occ_rec") or 0.0)
        u = np.asarray(u, dtype=float)
        ok = u > -1e8
        if beta == 0.0 or not ok.any():
            return self._occ_pool(u, ws)
        z = np.full(len(u), -1e9)
        z[ok] = (u[ok] - u[ok].mean()) / (u[ok].std() + 1e-6) + beta * np.arange(len(u))[ok] / max(1, len(u))
        last = {}                                    # the option-list (last) occurrence of a repeated word gets no
        cnt = {}                                     # recency bonus: recency ranks the in-text mentions only
        for i, w in enumerate(ws[:len(u)]):
            cnt[w] = cnt.get(w, 0) + 1; last[w] = i
        for w, i in last.items():
            if cnt[w] > 1 and ok[i]:
                z[i] -= beta * i / max(1, len(u))
        return self._occ_pool(z, ws)

    @staticmethod
    def _occ_pool(v, ws):''')
    a = sub(a, '''            u = self._occ_pool(u, s.obs_words)
            ok = u > -1e8''', '''            u = self._occ_pool_rec(u, s.obs_words)           # OCCREC_PATCH
            ok = u > -1e8''')
    a = sub(a, '''                u = self._occ_pool(u, s.obs_words)
                if occ == 2:''', '''                u = self._occ_pool_rec(u, s.obs_words)       # OCCREC_PATCH
                if occ == 2:''')
    open(A, "w").write(a)

m = open(Mu).read()
if "OCCREC_PATCH" not in m:
    m = sub(m, '''if a in ("hops", "carry", "occ", "bmask", "sep_reset", "dedup", "gate_alpha") else''',
            '''if a in ("hops", "carry", "occ", "bmask", "sep_reset", "dedup", "gate_alpha", "occ_rec") else''')
    m = sub(m, '''    if k == "dedup":
        r.rules["dedup"]''', '''    if k == "occ_rec":                                        # OCCREC_PATCH
        L = [0.0, 0.25, 0.5, 1.0, 2.0]
        cur = float(r.rules.get("occ_rec") or 0.0)
        i = min(range(len(L)), key=lambda j: abs(L[j] - cur))
        r.rules["occ_rec"] = float(w) if spec else L[max(0, min(len(L) - 1, i + rng.choice([-1, 1])))]
        return f"recency-aware occurrence pooling -> {r.rules['occ_rec']}"
    if k == "dedup":
        r.rules["dedup"]''')
    m = sub(m, '''"bmask", "sep_reset", "dedup"])   # DEDUP_PATCH''', '''"bmask", "sep_reset", "dedup", "occ_rec"])   # DEDUP_PATCH OCCREC_PATCH''')
    open(Mu, "w").write(m)
print("occrec patch applied to", root)
