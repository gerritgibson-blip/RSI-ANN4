"""Patch: learned contextual gate over argument-selection components (gene rules["gate"]=1). Idempotent.

Stage-0 lab (N=4096) showed that on instr the unary pointer alone picks the right name 95% of the time
('do not', 'no wait'), but the final choice is far worse: attention / identity-key / bridge terms are added with
fixed per-action weights, and facts, toolpick and instr all share the 'answer' action, so the weights that
make facts and toolpick work inject noise into instr.
Gate: score(w) = sum_c Z_c(w) * g_c(s), components Z = [unary z, attention z, identity-key raw, bridge raw],
g_c(s) = linear in s = [R-projection of the decision state (128), action one-hot, 1].
Fitted in closed form (ridge on per-candidate rows, target 1 for the oracle's argument words) on a short extra
pass of FRESH oracle episodes after the heads are fitted (so the unary scores are not over-confident).
Usage: python apply_gate.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
A, Ev, Mu = (os.path.join(root, "trn", f) for f in ("agent.py", "evolve.py", "mutate.py"))


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


a = open(A).read()
if "GATE_PATCH" not in a:
    a = sub(a, "MAXW = 192          # max words per observation kept for attention\n", '''MAXW = 192          # max words per observation kept for attention
NCOMP = 4           # GATE_PATCH: unary, attention, identity key, bridge


class GateFit:
    """ridge on per-candidate rows: x = outer(Z(w), [s, onehot(action), 1]); y = 1 for argument words."""
    def __init__(self, ds, dev="cuda"):
        self.P = NCOMP * ds
        self.XtX = torch.zeros(self.P, self.P, dtype=torch.float64, device=dev)
        self.XtY = torch.zeros(self.P, dtype=torch.float64, device=dev)
        self.Xs = torch.zeros(self.P, dtype=torch.float64, device=dev)
        self.Ys, self.n, self.ds, self.dev = 0.0, 0, ds, dev

    def add(self, Z, sv, y):
        """one decision: rows centred within the decision (only the ranking matters) and weighted so that every
        decision counts once, whatever the number of candidate words."""
        X = (torch.as_tensor(Z, dtype=torch.float64, device=self.dev)[:, :, None] * sv.double()[None, None, :]).reshape(len(Z), -1)
        y = torch.as_tensor(y, dtype=torch.float64, device=self.dev)
        X = X - X.mean(0, keepdim=True); y = y - y.mean()
        w = 1.0 / len(y)
        self.XtX += w * (X.T @ X); self.XtY += w * (X.T @ y); self.n += 1

    def solve(self, alpha=1e-2):
        n = max(1, self.n)
        C = self.XtX / n
        sd = C.diagonal().clamp_min(1e-12).sqrt()
        Cn = C / torch.outer(sd, sd)
        Cn.diagonal().add_(alpha)
        b = (self.XtY / n) / sd
        w = torch.linalg.solve(Cn, b) / sd
        return w.float().reshape(NCOMP, self.ds)
''')
    # decision state summary for the gate
    a = sub(a, '''                Q = self._head("q", Fd) if (pred is not None and self.attn is not None) else None
''', '''                Q = self._head("q", Fd) if (pred is not None and self.attn is not None) else None
                gate_on = self.attn is not None and ("gate" in self.heads or (collect is not None and "gate" in collect))
                if gate_on:
                    Sd = Fd.float() @ self.attn["R"]
                    Sd = Sd / (Sd.norm(dim=1, keepdim=True) + 1e-6)
''')
    a = sub(a, '''                    use_oracle = mode == "oracle" or pred is None or (mode == "dagger" and rng.random() < beta)
                    if use_oracle:
                        a, args = s.oracle
''', '''                    use_oracle = mode in ("oracle", "gate") or pred is None or (mode == "dagger" and rng.random() < beta)
                    if use_oracle:
                        a, args = s.oracle
                        if mode == "gate" and gate_on and a != "done" and (self._mix(a) > 0 or self._bindw(a) > 0):
                            Z = self._components(keys[r], Q[r], s, A2I[a])
                            n_ = len(Z)
                            y = [1.0 if s.obs_words[i] in args else 0.0 for i in range(n_)]
                            if any(y):
                                collect["gate"].add(Z, self._gstate(Sd[r], A2I[a]), y)
''')
    a = sub(a, '''                            sc = self._attn_scores(keys[r], Q[r], s, ai)''',
            '''                            sc = self._gated(keys[r], Q[r], s, ai, Sd[r]) if "gate" in self.heads else self._attn_scores(keys[r], Q[r], s, ai)''')
    a = sub(a, '''    def _bindw(self, a):''', '''    def _gstate(self, sd, ai):
        oh = torch.zeros(len(ACTIONS) + 1, device=sd.device)
        oh[ai] = 1.0; oh[-1] = 1.0
        return torch.cat([sd, oh])

    def _components(self, k, q, s, ai):
        """(n, NCOMP) per-word component scores: unary z, attention z, identity-key raw, bridge raw."""
        K2 = 2 * self.attn["R"].shape[1]
        att = (k[:, :K2] @ q[:K2]).cpu().numpy()
        n = len(att)
        kid = (k[:, K2:] @ q[K2:]).cpu().numpy() if k.shape[1] > K2 else np.zeros(n)
        u = np.array([float(s.sel[w][ai]) if w in s.sel else np.nan for w in range(n)])
        ok = ~np.isnan(u)
        uz = np.full(n, -4.0)
        if ok.any():
            uz[ok] = np.clip((u[ok] - u[ok].mean()) / (u[ok].std() + 1e-6), -4, 6)
        az = (att - att.mean()) / (att.std() + 1e-6)
        kz = (kid - kid.mean()) / (att.std() + 1e-6)          # same scale as the attention logits: keeps its confidence
        bs = self._bind_scores(s, n) if self.bind else np.zeros(n)
        return np.stack([uz, az, kz, bs], 1)

    def _gated(self, k, q, s, ai, sd):
        Z = self._components(k, q, s, ai)
        g = (self.heads["gate"] @ self._gstate(sd, ai)).cpu().numpy()          # (NCOMP,)
        return list(Z @ g)

    def _bindw(self, a):''')
    open(A, "w").write(a)

e = open(Ev).read()
if "GATE_PATCH" not in e:
    e = sub(e, '''    heads, val = fit_heads(acc)
    if MAX_ROUNDS == 0:
        del acc
        torch.cuda.empty_cache()''', '''    heads, val = fit_heads(acc)
    if MAX_ROUNDS == 0:
        del acc
        torch.cuda.empty_cache()
        if cfg is not None and res.rules.get("gate"):        # GATE_PATCH: contextual component gate, fresh episodes
            from .agent import GateFit
            gf = GateFit(cfg["R"].shape[1] + len(ACTIONS) + 1)
            run.heads = heads
            run.run(sample_specs({f: 1.0 for f in fams}, max(4, EP_ROUND0 // 8), seed + 2), mode="gate", collect={"gate": gf}, n_lm=0, B=B, seed=seed + 2)
            heads["gate"] = gf.solve()
            val["gate_rows"] = gf.n''')
    open(Ev, "w").write(e)

m = open(Mu).read()
if "GATE_PATCH" not in m:
    m = sub(m, '''    if k == "ptr_all":''', '''    if k == "gate":                                           # GATE_PATCH
        r.rules["gate"] = 0 if r.rules.get("gate") else 1
        return f"readout contextual gate -> {r.rules['gate']}"
    if k == "ptr_all":''')
    m = sub(m, '''        k = rng.choice(["bind", "bind", "key_id", "key_mix", "key_mix", "ptr_all"])''',
            '''        k = rng.choice(["bind", "bind", "key_id", "key_mix", "key_mix", "ptr_all", "gate"])''')
    m = sub(m, '''                spec = ("ptr_all", None, None) if a == "ptr_all" else''',
            '''                spec = ("gate", None, None) if a == "gate" else ("ptr_all", None, None) if a == "ptr_all" else''')
    open(Mu, "w").write(m)
print("gate patch applied to", root)
