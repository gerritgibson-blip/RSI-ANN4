"""patch: lean features (gene rules['chain_vq']=0 drops the membrane feature of delay-line relay neurons)
+ D-aware hardware check so the network can grow into the freed memory + forced form rule:<k>=<v>.
Idempotent; optional arg = package dir."""
import os, sys
os.chdir(sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN/trn")
def rep(s, a, b, n=1):
    assert s.count(a) == n, (a, s.count(a)); return s.replace(a, b)
s = open("reservoir.py").read()
if "def prep_features" in s:
    print("already applied"); raise SystemExit
s = rep(s, '''    def init_state(self, B):
''', '''    def prep_features(self):
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
''')
s = rep(s, '''        vq = ((st["v"] * 64) // self.theta).clamp_(-127, 127).to(torch.int8)
''', '''        vq = ((st["v"] * 64) // self.theta).clamp_(-127, 127).to(torch.int8)
        if getattr(self, "_vsel", None) is not None:      # relay membranes are near-copies of their neighbours' traces
            vq = vq[:, self._vsel]
''')
s = rep(s, '''        return 2 * self.N + (''', '''        nv = self.N if self.rules.get("chain_vq", 1) else -(-int((~self.is_chain).sum()) // 8) * 8
        return self.N + nv + (''')
open("reservoir.py", "w").write(s)

s = open("mutate.py").read()
s = rep(s, '''        if "grow" in ops and not (grow and fits(r.N + grow, r.mod["H"] if r.mod else 0)):''',
           '''        if "grow" in ops and not (grow and fits(r.N + grow, r.mod["H"] if r.mod else 0, r.D + 2 * grow)):''')
s = rep(s, '''    fits = fits or (lambda N, H: True)''', '''    fits = fits or (lambda N, H, D=None: True)''')
s = rep(s, '''    while not fits(r.N, r.mod["H"] if r.mod else 0):''', '''    while not fits(r.N, r.mod["H"] if r.mod else 0, r.D):''')
s = rep(s, '''        elif op.startswith("module:"):''', '''        elif op.startswith("rule:"):                 # forced form rule:<key>=<int>
            k, v = op.split(":", 1)[1].split("=")
            r.rules[k] = int(v)
            log.append(f"rule {k} -> {v}")
        elif op.startswith("module:"):''')
s = rep(s, '''"chain_max", "g_fix", "g_fix"])''', '''"chain_max", "g_fix", "g_fix", "chain_vq"])''')
s = rep(s, '''    elif k == "g_fix":''', '''    elif k == "chain_vq":  # expose relay-neuron membranes to the readout (1) or only their traces (0)
        R["chain_vq"] = 1 - R.get("chain_vq", 1)
    elif k == "g_fix":''')
open("mutate.py", "w").write(s)

s = open("evolve.py").read()
s = rep(s, '''def fits_hardware(N, H=0):''', '''def fits_hardware(N, H=0, D=None):''')
s = rep(s, '''    D = 2 * N + 3 * H          # 3H''', '''    D = D or 2 * N + 3 * H     # 3H''')
s = rep(s, '''    fits = lambda N, H: fits_hardware(N, H)''', '''    fits = lambda N, H, D=None: fits_hardware(N, H, D)''')
s = rep(s, '''        accept = mheads is not None and delta > 0 and z >= ACCEPT_Z''', '''        thr = st.pop("force_accept_z", None)          # one-shot override (operator-directed change, e.g. growth)
        accept = mheads is not None and ((delta > 0 and z >= ACCEPT_Z) if thr is None else z >= thr)''')
open("evolve.py", "w").write(s)
print("applied")
