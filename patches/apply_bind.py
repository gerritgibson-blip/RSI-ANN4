"""Patch: relational binding + word-identity keys for argument selection (readout-level, genes in res.rules).

Failure analysis of the N=15872 champion (gen 461) showed the remaining errors are binding errors:
  facts    : 95% of errors answer a value of the WRONG entity; success 11-29% when the answer is in the
             first fact vs 73-84% in the last (it mostly picks the most recent value)
  toolpick : success depends on the answer's position in the option list (23% first .. 78% fourth):
             the request-class x option-word conjunction is not being made
Two mechanisms (both pure readouts, nothing feeds back into the reservoir):
  bind_mix {action: w} : two-hop 'bridge' attention. A new pointer channel ('focus', trained on the words
      that link the question to the answer's clause, e.g. the package name in 'what is the version of X')
      picks a word; its hashed identity signature is the query, matched against each candidate's
      clause-context signature (sum of the signatures of the other words in its clause). Exact identity
      matching generalises to never-seen names (like a transformer's copy of a token's value vector).
  key_id w : the attention keys get the word's own identity signature (weight w relative to the
      reservoir part) so the query head can learn class -> option-word mappings (toolpick).
Usage: python apply_bind.py <dir containing trn/>   (idempotent)
"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
A = os.path.join(root, "trn", "agent.py")
E = os.path.join(root, "trn", "evolve.py")
Mu = os.path.join(root, "trn", "mutate.py")


def sub(src, old, new, count=1):
    if new in src:
        return src
    assert src.count(old) == count, (old, src.count(old))
    return src.replace(old, new)


s = open(A).read()
if "BIND_PATCH" not in s:
    s = sub(s, "import random, torch\n", "import random, torch, hashlib   # BIND_PATCH\n")
    s = sub(s, '"lm_off", "idle", "clause")', '"lm_off", "idle", "clause", "flab")')
    s = sub(s, "MAXW = 192          # max words per observation kept for attention\n", '''MAXW = 192          # max words per observation kept for attention
SIG_S = 128
_SIG = {}


def sig(w):
    """fixed random +-1 identity signature of a word (hash-seeded: defined for never-seen words)."""
    v = _SIG.get(w)
    if v is None:
        g = torch.Generator().manual_seed(int(hashlib.md5(w.encode()).hexdigest()[:8], 16))
        v = _SIG[w] = (torch.randint(0, 2, (SIG_S,), generator=g) * 2 - 1).float() / SIG_S ** 0.5
    return v


def clause_starts(ws):
    out, st = [0] * len(ws), 0
    for i, w in enumerate(ws):
        out[i] = st
        if w in PUNCT:
            st = i + 1
    return out


def focus_labels(ws, args):
    """pointer-style labels for the 'bridge' words: words of the answer's clause that also occur outside it
    in exactly one other clause (e.g. the entity named in the question). Labelled at their other occurrence."""
    pos = arg_positions(ws, args) if args else None
    if not pos:
        return None
    ce, cs = clause_ends(ws), clause_starts(ws)
    span = set()
    for p in pos:
        span |= set(range(cs[p], ce[p] + 1))
    argset, tgt = set(args), set()
    for i in span:
        v = ws[i]
        if v in PUNCT or v in argset:
            continue
        others = [j for j, w in enumerate(ws) if w == v and j not in span]
        if others and len({ce[j] for j in others}) == 1:
            tgt.update(others)
    labels = [0] * len(ws)
    for i in tgt:
        j = i
        while j < len(ws) - 1 and ws[j] not in PUNCT and j - i < MAXPTR - 1:
            j += 1
        labels[j] |= 1 << (j - i)
    return labels
''')
    s = sub(s, "        self.res, self.heads, self.corpus, self.attn = res, heads or {}, corpus_ids, attn\n",
            "        self.res, self.heads, self.corpus, self.attn = res, heads or {}, corpus_ids, attn\n"
            "        self.bind = bool(attn and any(v > 0 for v in (attn.get(\"bind\") or {}).values()))\n"
            "        self.nch = len(ACTIONS) + (1 if self.bind else 0)          # + focus channel\n")
    s = sub(s, "        s.labels = pointer_labels(s.obs_words, s.oracle[1]) if s.oracle else None\n",
            "        s.labels = pointer_labels(s.obs_words, s.oracle[1]) if s.oracle else None\n"
            "        s.flab = focus_labels(s.obs_words, s.oracle[1]) if (self.bind and s.oracle) else None\n")
    s = sub(s, "        NC = MAXPTR * len(ACTIONS)\n", "        NC = MAXPTR * self.nch\n")
    s = sub(s, """                            for k in range(MAXPTR):
                                if (lab[wi] >> k) & 1:
                                    ri.append(row); ci.append(c + k)
""", """                            for k in range(MAXPTR):
                                if (lab[wi] >> k) & 1:
                                    ri.append(row); ci.append(c + k)
                            fl = streams[b].flab
                            if fl is not None and wi < len(fl):
                                for k in range(MAXPTR):
                                    if (fl[wi] >> k) & 1:
                                        ri.append(row); ci.append(len(ACTIONS) * MAXPTR + k)
""")
    s = sub(s, "T = torch.zeros(len(drows), 2 * K, device=dev)", "T = torch.zeros(len(drows), keys[0].shape[1] if keys else 2 * K, device=dev)")
    s = sub(s, "view(len(wrows), len(ACTIONS), MAXPTR)", "view(len(wrows), self.nch, MAXPTR)")
    s = sub(s, 'if self.attn is not None and a != "done" and self._mix(a) > 0:',
            'if self.attn is not None and a != "done" and (self._mix(a) > 0 or self._bindw(a) > 0):')
    s = sub(s, """        k = torch.cat([own, cl], 1)
        return k - k.mean(0, keepdim=True)
""", """        k = torch.cat([own, cl], 1)
        kid = self.attn.get("key_id", 0) or 0
        if kid > 0:                                           # + the word's own identity signature
            S = torch.stack([sig(w) for w in s.obs_words[:n]]).to(k.device)
            S = S - S.mean(0, keepdim=True)
            k = k - k.mean(0, keepdim=True)
            scale = kid * k.norm(dim=1).mean() / (S.norm(dim=1).mean() + 1e-6)
            return torch.cat([k, scale * S], 1)
        return k - k.mean(0, keepdim=True)

    def _bindw(self, a):
        return (self.attn.get("bind") or {}).get(a, 0.0) if self.attn is not None else 0.0

    def _bind_scores(self, s, n):
        \"\"\"two-hop bridge attention: focus-pointer-weighted identity query . clause-context signatures.\"\"\"
        ws = s.obs_words[:n]
        S = torch.stack([sig(w) for w in ws])
        np_ = torch.tensor([0.0 if w in PUNCT else 1.0 for w in ws])[:, None]
        Sm = S * np_
        cum = torch.cat([torch.zeros(1, S.shape[1]), Sm.cumsum(0)])
        ce = torch.tensor([min(c, n - 1) for c in s.clause[:n]])
        cs = torch.tensor(clause_starts(ws))
        ctx = cum[ce + 1] - cum[cs] - Sm
        fc = len(ACTIONS)
        f = np.array([float(s.sel[w][fc]) if (w in s.sel and len(s.sel[w]) > fc) else -1e9 for w in range(n)])
        ok = f > -1e8
        if not ok.any() or f[ok].max() < 0.06:         # no confident bridge word: no binding term
            return np.zeros(n)
        conf = min(1.0, float(f[ok].max()) / 0.15)      # pointer outputs are small (sparse labels): scale, don't cut
        z = np.where(ok, (f - f[ok].mean()) / (f[ok].std() + 1e-6), -1e9)
        p = np.exp(4.0 * (z - z.max()))
        p /= p.sum()
        q = torch.from_numpy(p).float() @ S
        return conf * (ctx @ q).numpy()
""")
    s = sub(s, """    def _attn_scores(self, k, q, s, ai):
        a = (k @ q).cpu().numpy()
""", """    def _attn_scores(self, k, q, s, ai):
        K2 = 2 * self.attn["R"].shape[1]
        if (self.attn.get("key_id", 0) or 0) > 0 and k.shape[1] > K2:     # identity part weighted per action
            wm = (self.attn.get("key_mix") or {"answer": 1.0}).get(ACTIONS[ai], 0.0)
            a = (k[:, :K2] @ q[:K2] + wm * (k[:, K2:] @ q[K2:])).cpu().numpy()
        else:
            a = (k @ q).cpu().numpy()
""")
    s = sub(s, """            a = lam * a + u2
        return a.tolist()
""", """            a = lam * a + u2
        bw = self._bindw(ACTIONS[ai])
        if bw > 0:
            bs = self._bind_scores(s, len(a))
            a = np.asarray(a) + 3.0 * bw * (bs - np.median(bs))     # identity match ~1 x confidence, noise ~0.1
        return list(a)
""")
    open(A, "w").write(s)

e = open(E).read()
if "BIND_PATCH" not in e:
    e = sub(e, """    mix = res.rules.get("attn_mix") or {}
    if not any(v > 0 for v in mix.values()):
        return None
""", """    mix = res.rules.get("attn_mix") or {}       # BIND_PATCH
    bind = res.rules.get("bind_mix") or {}
    if not any(v > 0 for v in list(mix.values()) + list(bind.values())):
        return None
""")
    e = sub(e, '    return {"R": _ATTN[key], "mix": mix}\n',
            '    return {"R": _ATTN[key], "mix": mix, "bind": bind, "key_id": res.rules.get("key_id", 0), "key_mix": res.rules.get("key_mix")}\n\n\n'
            'def key_dim(cfg):\n'
            '    from .agent import SIG_S\n'
            '    return 2 * cfg["R"].shape[1] + (SIG_S if cfg.get("key_id") else 0)\n\n\n'
            'def n_ptr(cfg):\n'
            '    return MAXPTR * (len(ACTIONS) + (1 if cfg and any(v > 0 for v in (cfg.get("bind") or {}).values()) else 0))\n')
    e = sub(e, '"ptr": Ridge(D, MAXPTR * len(ACTIONS)),', '"ptr": Ridge(D, n_ptr(cfg)),')
    e = sub(e, '(2 * cfg["R"].shape[1] if cfg else 0)', '(key_dim(cfg) if cfg else 0)')
    open(E, "w").write(e)

m = open(Mu).read()
if "BIND_PATCH" not in m:
    m = sub(m, "def subpop(r, rng):\n", '''def bind(r, rng, spec=None):
    """BIND_PATCH: readout genes of the binding attention (changes head shapes -> not a reuse-able readout op)."""
    if spec:
        k, a, w = spec
    else:
        k = rng.choice(["bind", "bind", "key_id", "key_mix", "key_mix"])
        a = rng.choice(ATTN_ACTIONS) if k in ("bind", "key_mix") else None
    if k == "key_mix":
        mix = dict(r.rules.get("key_mix") or {"answer": 1.0})
        cur = float(mix.get(a, 0.0))
        w = w if spec else rng.choice([x for x in (0.0, 0.5, 1.0, 2.0) if x != cur])
        mix[a] = float(w)
        r.rules["key_mix"] = mix
        return f"readout key identity for {a} x{w}"
    if k == "key_id":
        cur = float(r.rules.get("key_id", 0) or 0)
        w = w if spec else rng.choice([x for x in (0.0, 0.5, 1.0, 2.0) if x != cur])
        r.rules["key_id"] = float(w)
        return f"readout key identity x{w}"
    mix = dict(r.rules.get("bind_mix") or {})
    cur = float(mix.get(a, 0.0))
    w = w if spec else rng.choice([x for x in (0.0, 0.5, 1.0, 2.0, 4.0) if x != cur])
    mix[a] = float(w)
    r.rules["bind_mix"] = mix
    return f"readout bridge attention {a} x{w}"


def subpop(r, rng):
''')
    m = sub(m, '"grow": 1, "readout": 2}', '"grow": 1, "readout": 2, "bind": 1.5}')
    m = sub(m, """        elif op.startswith("grow:"):                 # forced form grow:<n neurons>
""", """        elif op.startswith("bind"):                  # forced forms bind:<action>=<w>, bind:key_id=<w>
            spec = None
            if ":" in op:
                a, w = op.split(":", 1)[1].split("=")
                spec = ("key_id", None, float(w)) if a == "key_id" else (("key_mix", a[4:], float(w)) if a.startswith("kid_") else ("bind", a, float(w)))
            log.append(bind(r, rng, spec))
        elif op.startswith("grow:"):                 # forced form grow:<n neurons>
""")
    m = sub(m, 'if all(o.startswith("readout") for o in ops):', 'if all(o.startswith(("readout", "bind")) for o in ops):')
    open(Mu, "w").write(m)

# per-column-group regularisation: the focus channel must not change the alpha of the argument pointers
R_ = os.path.join(root, "trn", "readout.py")
r_ = open(R_).read()
if "BIND_PATCH" not in r_:
    r_ = sub(r_, '''    def fit(self, metric, alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
        """metric(pred, Y) -> higher is better, evaluated on validation rows."""''', '''    def fit(self, metric, alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), groups=None):
        """metric(pred, Y) -> higher is better, evaluated on validation rows.
        groups (BIND_PATCH): list of column slices, each gets its own alpha (score = first group's)."""
        if groups and self.vX:
            self.flush()
            self._to_host()
            vX, vY = torch.cat(self.vX), torch.cat(self.vY)
            best = [(-1e9, None) for _ in groups]
            Wout = None
            for a in alphas:
                W = self._solve(a)
                if W is None:
                    continue
                P = self.predict(vX, W)
                if Wout is None:
                    Wout = W.clone()
                for gi, g in enumerate(groups):
                    sc = metric(P[:, g], vY[:, g])
                    if sc > best[gi][0]:
                        best[gi] = (sc, a)
                        Wout[:, g] = W[:, g]
                del W, P
            self.val_score, self.alpha, self.W = best[0][0], best[0][1], Wout
            return best[0][0]''')
    open(R_, "w").write(r_)
e = open(E).read()
if "groups=" not in e:
    e = sub(e, '''        val[name] = round(acc[name].fit(metric), 4)''', '''        NP = MAXPTR * len(ACTIONS)
        grp = [slice(0, NP), slice(NP, acc[name].C)] if (name == "ptr" and acc[name].C > NP) else None
        val[name] = round(acc[name].fit(metric, groups=grp), 4)''')
    open(E, "w").write(e)
print("bind patch applied to", root)
