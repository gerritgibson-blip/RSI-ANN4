"""Quantized ternary reservoir.

Hardware-efficient by construction:
  * synapses W[i, j] (i -> j) are ternary {-1, 0, +1}, stored as int8;
  * a tick is one int8 tensor-core matmul (spikes x W -> int32) plus integer shifts;
  * membrane potentials are int16, spikes are ternary int8, readout traces int8.

Dynamics per tick (iterated once per timestep):
  v  <- v - leak(v) + g_rec * (s @ W) + g_in * Win[token]
  s  <- +1 if v > theta, -1 if v < -theta, else 0       (ternary spike)
  v  <- v - s * theta                                   (subtractive reset)
  tr <- tr - leak3(tr) + 16 * s                         (int8 readout trace)
leak(v) = sign(v) * max(1, |v| >> k_i): passive decay that always reaches 0, with
a per-neuron shift k_i, so the reservoir has many timescales. Token 0 (PAD) drives
nothing, so between inputs activity echoes and passively decays. g_rec is tuned so
that echoes die out after ~`echo_target` ticks (echo-state property is enforced).
"""
import math, torch

CHARS = " abcdefghijklmnopqrstuvwxyz0123456789.,;:?!'->"
PAD, ACT, SEP = 0, 1, 2
VOCAB = 3 + len(CHARS)
C2I = {c: i + 3 for i, c in enumerate(CHARS)}
I2C = {i: c for c, i in C2I.items()}


SPACE = C2I[" "]
P31 = 2 ** 31 - 1


def encode(text):
    return [C2I[c] for c in text if c in C2I]


def _sgn_decay(x, k):
    a = x.abs()
    d = torch.maximum((a >> k), torch.ones_like(a)) * (a > 0)
    return x - torch.sign(x) * d


class Reservoir:
    def __init__(self, N=4096, fan_in=64, in_frac=0.25, in_fan=None, seed=0, device="cuda", kmax=8, chain_frac=0.6):
        assert N % 256 == 0
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.device, self.N = device, N
        self.W = self._rand_block(N, N, fan_in, g)
        self.k = torch.randint(1, kmax + 1, (N,), generator=g).to(torch.int32)           # leak shifts: tau ~ 2^k ticks
        self.theta = torch.full((N,), 64, dtype=torch.int32)
        self.is_in = torch.rand(N, generator=g) < in_frac                          # input nodes
        self.in_fan = in_fan or max(16, N // 16)
        self.rules = {"echo_target": 200, "think": 4, "trace_shift": 3, "fan_in": fan_in, "target_rate": 0.06, "chain_max": 128}
        self.pred = torch.full((N,), N, dtype=torch.long)     # delay-chain predecessor (N = none)
        self.is_chain = torch.zeros(N, dtype=torch.bool)
        self._build_chains(torch.nonzero(self.is_in).flatten(), torch.nonzero(~self.is_in).flatten(), N // 64, int(N * chain_frac), g)
        self.Win = self._rand_win(VOCAB, g)
        self.g_rec, self.g_in = 16, 160
        self.mod = None
        self.to(device)

    # ---------- construction helpers ----------
    def _rand_block(self, rows, cols, fan_in, g):
        """each column (post-synaptic neuron) receives `fan_in` random ternary inputs."""
        W = torch.zeros(rows, cols, dtype=torch.int8)
        idx = torch.randint(0, rows, (fan_in, cols), generator=g)
        sgn = (torch.randint(0, 2, (fan_in, cols), generator=g) * 2 - 1).to(torch.int8)
        W.scatter_(0, idx, sgn)
        return W

    def _build_chains(self, heads_pool, body_pool, H, n_body, g):
        """axonal delay lines: input neuron -> c1 -> c2 -> ... ; a spike walks one neuron per tick and
        falls off the end, so the reservoir holds an ordered, passively vanishing echo of its input."""
        body = body_pool[torch.randperm(len(body_pool), generator=g)[:n_body]].tolist()
        heads = heads_pool[torch.randperm(len(heads_pool), generator=g)].tolist()
        Lmax = self.rules["chain_max"]
        while body and heads:
            L = int(torch.randint(8, Lmax + 1, (1,), generator=g))
            seg, body = body[:L], body[L:]
            prev = heads.pop()
            for n in seg:
                self.pred[n] = prev; self.is_chain[n] = True; prev = n
        body = torch.nonzero(self.is_chain).flatten()
        self.W[:, body] = 0                      # relay neurons listen only to their predecessor ...
        self.k[body] = 1                         # ... and forget fast; their outputs still feed the mixing pool

    def _rand_win(self, V, g):
        Win = torch.zeros(V, self.N, dtype=torch.int8)
        pool = torch.nonzero(self.is_in.cpu()).flatten()
        for t in range(1, V):          # PAD (0) has no input
            sel = pool[torch.randperm(len(pool), generator=g)[: self.in_fan]]
            Win[t, sel] = (torch.randint(0, 2, (len(sel),), generator=g) * 2 - 1).to(torch.int8)
        return Win

    # ---------- word-match module (fast associative memory) ----------
    def add_module(self, H=512, decay=6, gm=None, fan=8, seed=0, ctx_pool="mixing", bow=0):
        """content-addressable word memory: every finished word gets a hashed ternary signature `sig`;
        the module stores sig (x) ctx (ctx = ternary snapshot of H reservoir traces) in a per-stream
        fast-weight matrix M that passively decays per word, and queries M with the new word first:
        r = sig @ M = the contexts in which this word was seen before (0 for a novel word).
        r of the last two words is exposed to the readouts and injected into input nodes (so the
        delay lines carry it forward). Gives the reservoir word identity matching / binding."""
        g = torch.Generator().manual_seed(seed)
        cand = torch.nonzero((self.is_chain if ctx_pool == "chain" else ~self.is_chain).cpu()).flatten()
        inn = torch.nonzero(self.is_in.cpu()).flatten()
        Wm = torch.zeros(H, self.N, dtype=torch.int8)
        idx = inn[torch.randint(0, len(inn), (H, fan), generator=g)]
        Wm.scatter_(1, idx, (torch.randint(0, 2, (H, fan), generator=g) * 2 - 1).to(torch.int8))
        self.mod = {"H": H, "decay": decay, "gm": gm if gm is not None else max(1, self.g_in // 64), "fan": fan, "ctx_pool": ctx_pool,
                    "bow": bow,      # >0: bag-of-recent-words features (decaying sum of word signatures, 2^-bow per word)
                    "ctx": cand[torch.randperm(len(cand), generator=g)[:H]],
                    "A": torch.randint(1, P31, (H,), generator=g, dtype=torch.int64),
                    "B": torch.randint(0, P31, (H,), generator=g, dtype=torch.int64), "Wm": Wm}
        self.to(self.device)

    def to(self, device):
        for n in ("W", "k", "theta", "is_in", "Win", "pred", "is_chain"):
            setattr(self, n, getattr(self, n).to(device))
        if getattr(self, "mod", None):
            for n in ("ctx", "A", "B", "Wm"):
                self.mod[n] = self.mod[n].to(device)
            self._Wmf = self.mod["Wm"].float()
        self.device = device
        return self

    # ---------- state / dynamics ----------
    def prep_features(self):
        """index of neurons whose membrane is exposed (all, or non-relay only when rules['chain_vq'] == 0).
        Called outside CUDA-graph capture (init_state); features() only reads it."""
        if self.rules.get("chain_vq", 1):
            self._vsel = None
            return
        idx = torch.nonzero(~self.is_chain).flatten()
        pad = (-len(idx)) % 8                                  # int8 tensor-core Gram needs D % 8 == 0
        self._vsel = torch.cat([idx, torch.nonzero(self.is_chain).flatten()[:pad]])

    def init_state(self, B):
        self.prep_features()
        z = lambda dt: torch.zeros(B, self.N, dtype=dt, device=self.device)
        st = {"v": z(torch.int32), "s": z(torch.int8), "tr": z(torch.int8)}
        if self.mod:
            H = self.mod["H"]
            st["h"] = torch.zeros(B, dtype=torch.int64, device=self.device)
            st["M"] = torch.zeros(B, H, H, device=self.device)      # stored as M_true * ic (lazy decay)
            st["ic"] = torch.ones(B, device=self.device)
            st["r0"] = torch.zeros(B, H, dtype=torch.int8, device=self.device)
            st["r1"] = torch.zeros(B, H, dtype=torch.int8, device=self.device)
            if self.mod.get("bow"):
                st["bw"] = torch.zeros(B, H, device=self.device)
        return st

    def _module_step(self, st, tokens, put):
        """masked over the whole batch (no host sync): streams that finish a word query and update M."""
        m = self.mod
        if self.rules.get("sep_reset"):                  # SEPRESET_PATCH: a new episode starts with an empty memory
            keep = (tokens != SEP).float()
            st["M"].mul_(keep[:, None, None])
            if "bw" in st:
                st["bw"].mul_(keep[:, None])
        sep = (tokens < 3) | (tokens == SPACE)
        h = st["h"]
        end = sep & (h != 0)
        sig = (((h[:, None] * m["A"] + m["B"]) % P31) % 3 - 1).float() * end[:, None]   # hashed ternary word signature
        new_h = torch.where(sep, torch.zeros_like(h), (h * 1000003 + tokens) % P31)
        M, ic = st["M"], st["ic"]
        r = torch.bmm(sig[:, None, :], M).squeeze(1) / ic[:, None]                     # query before storing
        ctx = torch.sign(st["tr"][:, m["ctx"]].float())
        new_ic = torch.where(end, ic / (1 - 2.0 ** -m["decay"]), ic)                    # decay = growth of the inverse scale
        M.baddbmm_((sig * (64 * new_ic)[:, None])[:, :, None], ctx[:, None, :])         # rank-1 store (0 where no word ended)
        put("ic", new_ic)
        rq = (r * (2.25 / m["H"])).round().clamp(-127, 127).to(torch.int8)
        endc = end[:, None]
        new_r1 = torch.where(endc, st["r0"], st["r1"])
        new_r0 = torch.where(endc, rq, st["r0"])
        put("h", new_h); put("r1", new_r1); put("r0", new_r0)
        if m.get("bow"):     # pooled word embedding of the recent words (order-free; the delay lines carry order)
            bw = st["bw"]
            put("bw", bw - bw * (2.0 ** -m["bow"]) * end[:, None] + sig)
        return (rq.float() @ self._Wmf).to(torch.int32) * m["gm"]       # rq is 0 where no word ended

    def renorm(self, st):
        """fold the lazy decay scale back into M (keeps the fp32 values bounded)."""
        if self.mod:
            st["M"].div_(st["ic"][:, None, None])
            st["ic"].fill_(1.0)

    def sparse_W(self):
        """W^T as a CSR matrix: each tick costs nnz x B instead of N^2 x B (W is ~0.2% dense)."""
        idx = torch.nonzero(self.W.T)
        vals = self.W.T[idx[:, 0], idx[:, 1]].float()
        return torch.sparse_coo_tensor(idx.T, vals, (self.N, self.N)).coalesce().to_sparse_csr()

    @torch.no_grad()
    def step(self, st, tokens, inplace=False, csr=None):
        """one tick. inplace=True writes into the existing state tensors (needed for CUDA-graph replay);
        csr = sparse_W() for the event-sparse recurrent product (exact: integer values in fp32)."""
        def put(k, val):
            if inplace:
                st[k].copy_(val)
            else:
                st[k] = val
        if self.g_rec == 0:                               # GREC0_PATCH: no recurrent drive -> skip the product (exact)
            I = 0
        elif csr is not None:
            I = (csr @ st["s"].T.contiguous().float()).T.contiguous().to(torch.int32)
        else:
            I = torch._int_mm(st["s"], self.W)                                       # int8 x int8 -> int32
        sx = torch.cat([st["s"], torch.zeros_like(st["s"][:, :1])], 1)
        relay = sx[:, self.pred].to(torch.int32) * (2 * self.theta)               # delay-chain hop
        v = _sgn_decay(st["v"], self.k) + self.g_rec * I + relay + self.g_in * self.Win[tokens].to(torch.int32)
        if self.mod:
            v = v + self._module_step(st, tokens, put)
        s = (v > self.theta).to(torch.int8) - (v < -self.theta).to(torch.int8)
        v = (v - s.to(torch.int32) * self.theta).clamp_(-32767, 32767)
        tr = _sgn_decay(st["tr"].to(torch.int32), self.rules["trace_shift"]) + 16 * s.to(torch.int32)
        put("v", v); put("s", s); put("tr", tr.clamp_(-127, 127).to(torch.int8))
        if not inplace:
            self.renorm(st)
        return st

    def graph(self, B):
        """CUDA-graph the whole tick (step + features): one launch per tick instead of ~40 kernels.
        Must be rebuilt whenever parameters change (mutation / tuning)."""
        return GraphStep(self, B)

    def features(self, st):
        """readout sees spike traces and the (passively decaying) membrane potentials, both int8."""
        vq = ((st["v"] * 64) // self.theta).clamp_(-127, 127).to(torch.int8)
        if getattr(self, "_vsel", None) is not None:      # relay membranes are near-copies of their neighbours' traces
            vq = vq[:, self._vsel]
        if self.mod:
            f = [st["tr"], vq, st["r0"], st["r1"]]
            if self.mod.get("bow"):
                f.append((st["bw"] * 16).round().clamp(-127, 127).to(torch.int8))
            return torch.cat(f, 1)
        return torch.cat([st["tr"], vq], 1)

    @property
    def D(self):
        nv = self.N if self.rules.get("chain_vq", 1) else -(-int((~self.is_chain).sum()) // 8) * 8
        return self.N + nv + ((2 + (1 if self.mod.get("bow") else 0)) * self.mod["H"] if self.mod else 0)

    # ---------- self-regulation: homeostasis + echo-state tuning ----------
    @torch.no_grad()
    def drive(self, text_ids, B=32, T=200, pad_after=0, seed=0):
        g = torch.Generator().manual_seed(seed)
        gs = self.graph(B)
        offs = torch.randint(0, len(text_ids) - T - 1, (B,), generator=g)
        seq = torch.stack([text_ids[o:o + T] for o in offs]).to(self.device)
        pad = torch.zeros(B, dtype=torch.long, device=self.device)
        rates, act = torch.zeros(self.N, device=self.device), []
        for t in range(T + pad_after):
            gs(seq[:, t] if t < T else pad)
            a = gs.st["s"].abs().float()
            if t < T:
                rates += a.mean(0)
            act.append(a.mean())
        return rates / T, torch.stack(act).tolist()

    def homeostasis(self, text_ids, iters=6):
        tgt = self.rules["target_rate"]
        for i in range(iters):
            r, _ = self.drive(text_ids, T=160, seed=100 + i)
            f = ((r + 1e-3) / tgt).clamp(0.5, 2.0) ** 0.5
            self.theta = (self.theta.float() * f).round().clamp(8, 4000).to(torch.int32)

    def echo_time(self, text_ids, seed=0):
        _, act = self.drive(text_ids, T=100, pad_after=600, seed=seed)
        a0 = sum(act[80:100]) / 20 + 1e-9
        for t, a in enumerate(act[100:]):
            if a < 0.02 * a0:
                return t
        return 10 ** 6   # self-sustaining: violates passive decay

    def tune(self, text_ids):
        """alternate homeostasis and a search over g_rec so echoes last ~echo_target ticks and then die."""
        tgt = self.rules["echo_target"]
        if self.rules.get("g_fix") is not None:        # evolved recurrent gain: homeostasis only, no echo search
            self.g_rec = int(self.rules["g_fix"])
            self.homeostasis(text_ids); self.homeostasis(text_ids)
            return self.echo_time(text_ids, seed=7)
        for _ in range(2):
            self.homeostasis(text_ids)
            lo, hi = 1, 256
            while lo < hi:                     # largest g_rec whose echo still dies within target
                mid = (lo + hi + 1) // 2
                self.g_rec = mid
                if self.echo_time(text_ids) <= tgt:
                    lo = mid
                else:
                    hi = mid - 1
            self.g_rec = lo
        self.homeostasis(text_ids)
        while self.g_rec > 1 and self.echo_time(text_ids, seed=7) > tgt:   # homeostasis may re-ignite: back off
            self.g_rec -= 1
        return self.echo_time(text_ids, seed=7)

    # ---------- persistence ----------
    def state_dict(self):
        return {"W": self.W.cpu(), "k": self.k.cpu(), "theta": self.theta.cpu(), "is_in": self.is_in.cpu(),
                "Win": self.Win.cpu(), "pred": self.pred.cpu(), "is_chain": self.is_chain.cpu(), "g_rec": self.g_rec, "g_in": self.g_in, "in_fan": self.in_fan, "rules": dict(self.rules), "N": self.N,
                "mod": ({k_: (v.cpu() if torch.is_tensor(v) else v) for k_, v in self.mod.items()} if self.mod else None)}

    @classmethod
    def from_state(cls, d, device="cuda"):
        r = cls.__new__(cls)
        for k_, v in d.items():
            setattr(r, k_, v)
        r.rules = dict(d["rules"])
        r.mod = dict(d["mod"]) if d.get("mod") else None
        r.device = "cpu"
        return r.to(device)

    def n_edges(self):
        return int((self.W != 0).sum())


class GraphStep:
    """static-buffer CUDA graph of Reservoir.step (sparse recurrent product) + features for batch B.
    gs(tokens) -> features (a view of a static buffer, overwritten by the next call: copy rows you keep).
    gs.st is the live state."""
    RENORM = 64

    def __init__(self, res, B):
        self.res, self.t = res, 0
        self.st = res.init_state(B)
        self.tok = torch.zeros(B, dtype=torch.long, device=res.device)
        self.cuda = str(res.device).startswith("cuda")
        if not self.cuda:
            return
        self.csr = res.sparse_W()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                res.step(self.st, self.tok, inplace=True, csr=self.csr); res.features(self.st)
        torch.cuda.current_stream().wait_stream(side)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            res.step(self.st, self.tok, inplace=True, csr=self.csr)
            self.F = res.features(self.st)
        self.reset()

    def reset(self):
        for v in self.st.values():
            v.zero_()
        if "ic" in self.st:
            self.st["ic"].fill_(1.0)

    @torch.no_grad()
    def __call__(self, tokens):
        if not self.cuda:
            self.res.step(self.st, tokens)
            return self.res.features(self.st)
        self.tok.copy_(tokens)
        self.g.replay()
        self.t += 1
        if self.t % self.RENORM == 0:
            self.res.renorm(self.st)
        return self.F
