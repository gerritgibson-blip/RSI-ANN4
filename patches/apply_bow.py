"""patch: bag-of-recent-words module features (idempotent: skips if already applied)."""
import os
import sys; os.chdir(sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN/trn")
def rep(s, a, b, n=1):
    assert s.count(a) == n, (a, s.count(a)); return s.replace(a, b)
s = open("reservoir.py").read()
if '"bow": bow' in s:
    print("already applied"); raise SystemExit
s=rep(s,'''    def add_module(self, H=512, decay=6, gm=None, fan=8, seed=0, ctx_pool="mixing"):''','''    def add_module(self, H=512, decay=6, gm=None, fan=8, seed=0, ctx_pool="mixing", bow=0):''')
s=rep(s,'''        self.mod = {"H": H, "decay": decay, "gm": gm if gm is not None else max(1, self.g_in // 64), "fan": fan, "ctx_pool": ctx_pool,''','''        self.mod = {"H": H, "decay": decay, "gm": gm if gm is not None else max(1, self.g_in // 64), "fan": fan, "ctx_pool": ctx_pool,
                    "bow": bow,      # >0: bag-of-recent-words features (decaying sum of word signatures, 2^-bow per word)''')
s=rep(s,'''            st["r1"] = torch.zeros(B, H, dtype=torch.int8, device=self.device)
        return st''','''            st["r1"] = torch.zeros(B, H, dtype=torch.int8, device=self.device)
            if self.mod.get("bow"):
                st["bw"] = torch.zeros(B, H, device=self.device)
        return st''')
s=rep(s,'''        put("h", new_h); put("r1", new_r1); put("r0", new_r0)
''','''        put("h", new_h); put("r1", new_r1); put("r0", new_r0)
        if m.get("bow"):     # pooled word embedding of the recent words (order-free; the delay lines carry order)
            bw = st["bw"]
            put("bw", bw - bw * (2.0 ** -m["bow"]) * end[:, None] + sig)
''')
s=rep(s,'''        if self.mod:
            return torch.cat([st["tr"], vq, st["r0"], st["r1"]], 1)''','''        if self.mod:
            f = [st["tr"], vq, st["r0"], st["r1"]]
            if self.mod.get("bow"):
                f.append((st["bw"] * 16).round().clamp(-127, 127).to(torch.int8))
            return torch.cat(f, 1)''')
s=rep(s,'''        return 2 * self.N + (2 * self.mod["H"] if self.mod else 0)''','''        return 2 * self.N + ((2 + (1 if self.mod.get("bow") else 0)) * self.mod["H"] if self.mod else 0)''')
open("reservoir.py","w").write(s)
s=open("mutate.py").read()
s=rep(s,'''    k = rng.choice(["decay", "gm", "ctx", "H", "fan", "ctx_pool", "remove"] if rng.random() < 0.1 else ["decay", "gm", "ctx", "H", "fan", "ctx_pool"])''','''    k = rng.choice(["decay", "gm", "ctx", "H", "fan", "ctx_pool", "remove"] if rng.random() < 0.1 else ["decay", "gm", "ctx", "H", "fan", "ctx_pool", "bow", "bow"])''')
s=rep(s,'''    if k == "gm":''','''    if k == "bow":        # bag-of-recent-words features: off, or a word-decay shift (memory of ~2^bow words)
        m["bow"] = rng.choice([x for x in (0, 2, 3, 4, 5) if x != m.get("bow", 0)])
        return f"module bag-of-words -> {m['bow']}"
    if k == "gm":''')
s=rep(s,'''    r.add_module(H=H, decay=keep["decay"], gm=keep["gm"], fan=int(fan), seed=rng.randrange(2 ** 31), ctx_pool=pool)''','''    r.add_module(H=H, decay=keep["decay"], gm=keep["gm"], fan=int(fan), seed=rng.randrange(2 ** 31), ctx_pool=pool, bow=m.get("bow", 0))''')
s=rep(s,'''        elif op == "module":
            log.append(module(r, rng))''','''        elif op.startswith("module:"):               # forced form module:bow=<shift>
            k, v = op.split(":", 1)[1].split("=")
            r.mod[k] = int(v)
            log.append(f"module {k} -> {v}")
        elif op == "module":
            log.append(module(r, rng))''')
open("mutate.py","w").write(s)
s=open("evolve.py").read()
s=rep(s,'''    D = 2 * N + 2 * H
    need''','''    D = 2 * N + 3 * H          # 3H: module r0, r1 + optional bag-of-words features
    need''')
s=rep(s,'''{"H": m["H"], "decay": m["decay"], "gm": m["gm"], "fan": m["fan"]}''','''{"H": m["H"], "decay": m["decay"], "gm": m["gm"], "fan": m["fan"], "ctx": m.get("ctx_pool"), "bow": m.get("bow", 0)}''')
open("evolve.py","w").write(s)
print("applied")
