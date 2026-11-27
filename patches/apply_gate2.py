"""Patch: gate v3 = gene rules["gate"] == 2 (GATE2_PATCH). Idempotent.

Probe (2026-11-27, champion gen 732): in 18/25 instr2 failures the unary pointer ranks the answer first and the
attention term overrides it; on toolpick2 more attention helps (gen 735: answer attention x2 -> toolpick2
0.90 -> 0.94 but instr2 0.95 -> 0.90). Both use the 'answer' action, so per-action weights cannot serve both:
the weighting must depend on the decision context -> the contextual gate. Gate v1/v2 failed (facts 98 -> 57%)
because (a) its ridge shrank all component weights toward ZERO, discarding the good unary signal whenever data
was thin, and (b) its scorer did not mask non-candidate words (the default scorer does) nor pool occurrences.
v3: ridge shrinks toward a scaled copy of the DEFAULT mix (unary 1, attention lam, identity key lam*wm, bridge
3*bw per action), so gate-off is the alpha -> inf limit and the fit learns only context corrections; the scorer
masks non-candidates and applies occ pooling exactly as the default path.
Usage: python apply_gate2.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
A, Ev, Mu = (os.path.join(root, "trn", f) for f in ("agent.py", "evolve.py", "mutate.py"))


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


a = open(A).read()
if "GATE2_PATCH" not in a:
    a = sub(a, '''import random, torch, hashlib, copy, collections   # BIND_PATCH REASON_PATCH''',
            '''import random, torch, hashlib, copy, collections, os   # BIND_PATCH REASON_PATCH GATE2_PATCH''')
    a = sub(a, '''    def solve(self, alpha=1e-2):
        n = max(1, self.n)
        C = self.XtX / n
        sd = C.diagonal().clamp_min(1e-12).sqrt()
        Cn = C / torch.outer(sd, sd)
        Cn.diagonal().add_(alpha)
        b = (self.XtY / n) / sd
        w = torch.linalg.solve(Cn, b) / sd''', '''    def solve(self, alpha=1e-2, w0=None):
        n = max(1, self.n)
        C = self.XtX / n
        sd = C.diagonal().clamp_min(1e-12).sqrt()
        Cn = C / torch.outer(sd, sd)
        Cn.diagonal().add_(alpha)
        b = (self.XtY / n) / sd
        if w0 is not None:                       # GATE2_PATCH: shrink toward c * w0 (the default mix), not zero
            w0 = torch.as_tensor(w0, dtype=torch.float64, device=C.device).reshape(-1)
            c = float((self.XtY / n) @ w0) / max(1e-12, float(w0 @ (C @ w0)))
            b = b + alpha * (sd * (max(c, 0.0) * w0))
        w = torch.linalg.solve(Cn, b) / sd''')
    a = sub(a, '''    def _gated(self, k, q, s, ai, sd):
        Z = self._components(k, q, s, ai)''', '''    def _gated(self, k, q, s, ai, sd):
        if int(self.res.rules.get("gate") or 0) == 2:            # GATE2_PATCH: masked, occ-pooled components
            Z = self._components2(k, q, s, ai)
            g = (self.heads["gate"] @ self._gstate(sd, ai)).cpu().numpy()
            sc = Z @ g
            ok = np.array([w in s.sel for w in range(len(sc))])
            return list(np.where(ok, sc, -1e9)) if ok.any() else list(sc)
        Z = self._components(k, q, s, ai)''')
    a = sub(a, '''                            Z = self._components(keys[r], Q[r], s, A2I[a])
                            n_ = len(Z)
                            y = [1.0 if s.obs_words[i] in args else 0.0 for i in range(n_)]''', '''                            Z = self._components(keys[r], Q[r], s, A2I[a])
                            n_ = len(Z)
                            y = [1.0 if s.obs_words[i] in args else 0.0 for i in range(n_)]
                            if int(self.res.rules.get("gate") or 0) == 2:      # GATE2_PATCH: candidates only, pooled
                                cand_ = [i for i in range(n_) if i in s.sel]
                                if len(cand_) >= 2:
                                    Z = self._components2(keys[r], Q[r], s, A2I[a])[cand_]
                                    y = [y[i] for i in cand_]
                                    hn_ = int(os.environ.get("TRN_GATE_HN", 4))   # hard negatives: the answer + the
                                    if hn_ > 0:                                    # top-k wrong words under the default mix
                                        gs_ = self._gstate(Sd[r], A2I[a])
                                        g0_ = torch.as_tensor(self.gate_prior(len(gs_)), dtype=torch.float32, device=gs_.device) @ gs_
                                        s0_ = Z @ g0_.cpu().numpy()
                                        neg_ = sorted((i for i in range(len(y)) if not y[i]), key=lambda i: -s0_[i])[:hn_]
                                        keep_ = [i for i in range(len(y)) if y[i]] + neg_
                                        Z = Z[keep_]; y = [y[i] for i in keep_]
                                else:
                                    y = [0.0]''')
    a = sub(a, '''    def _bindw(self, a):''', '''    def _components2(self, k, q, s, ai):
        """GATE2_PATCH: like _components, but the unary score is occurrence-pooled exactly as in _attn_scores."""
        Z = self._components(k, q, s, ai)
        occ = int(self.res.rules.get("occ") or 0)
        if occ and ACTIONS[ai] == "answer" and s.sel:
            n = Z.shape[0]
            u = np.array([float(s.sel[w][ai]) if w in s.sel else -1e9 for w in range(n)])
            u = self._occ_pool(u, s.obs_words)
            ok = u > -1e8
            uz = np.full(n, -4.0)
            if ok.any():
                uz[ok] = np.clip((u[ok] - u[ok].mean()) / (u[ok].std() + 1e-6), -4, 6)
            Z[:, 0] = uz
        return Z

    def gate_prior(self, ds):
        """GATE2_PATCH: (NCOMP, ds) gate weights reproducing the default per-action mix."""
        K = ds - len(ACTIONS) - 1
        W0 = np.zeros((NCOMP, ds))
        for ai, a in enumerate(ACTIONS):
            lam = self._mix(a)
            wm = (self.attn.get("key_mix") or {"answer": 1.0}).get(a, 0.0) if (self.attn.get("key_id") or 0) > 0 else 0.0
            W0[0, K + ai], W0[1, K + ai], W0[2, K + ai], W0[3, K + ai] = 1.0, lam, lam * wm, 3.0 * self._bindw(a)
        return W0

    def _bindw(self, a):''')
    open(A, "w").write(a)

e = open(Ev).read()
if "GATE2_PATCH" not in e:
    e = sub(e, '''            heads["gate"] = gf.solve()''', '''            if int(res.rules.get("gate") or 0) == 2:      # GATE2_PATCH: prior-centred fit
                heads["gate"] = gf.solve(GATE_ALPHA, w0=run.gate_prior(gf.ds))
            else:
                heads["gate"] = gf.solve()''')
    e = sub(e, '''TRAIN_EMPH = None''', '''GATE_ALPHA = float(os.environ.get("TRN_GATE_ALPHA", 1.0))   # GATE2_PATCH: shrinkage toward the default mix
TRAIN_EMPH = None''')
    open(Ev, "w").write(e)

m = open(Mu).read()
if "GATE2_PATCH" not in m:
    m = sub(m, '''        r.rules["gate"] = 0 if r.rules.get("gate") else 1''', '''        r.rules["gate"] = 0 if r.rules.get("gate") else 2      # GATE2_PATCH: v3 gate''')
    open(Mu, "w").write(m)
print("gate2 patch applied to", root)
