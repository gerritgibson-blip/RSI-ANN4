"""Patch: multi-hop binding (gene rules["hops"]=2) and goal memory via a self-written note (gene rules["carry"]=1).
Idempotent. Requires apply_bind (focus channel), apply_gate, apply_stage4 already applied.

hops: bridge labels come from a breadth-first chain over clauses linked by shared rare words (count == 2 in the
  observation): answer clause -> dependency line -> question. The focus pointer is trained on the chain's START
  (e.g. the package named in the question), and a new 3-way decision head 'hop' (fitted in the decision Gram)
  predicts the chain depth (0 = no binding, 1, 2). At decision time the bridge query is propagated h times:
  focus word -> clauses containing it -> their other rare words -> candidates whose clause contains those.
carry: a pointer channel selects words to carry; they are prepended to the next observation as 'note : w .'.
  Labels come from rolling the oracle forward on a copy of the environment: carry a word of the current
  observation if a FUTURE decision's answer clause contains it while it is absent elsewhere in that future
  observation (so only the note can supply the bridge). Training uses the oracle note, evaluation the predicted one.
Usage: python apply_reason.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
A, Ev, Mu = (os.path.join(root, "trn", f) for f in ("agent.py", "evolve.py", "mutate.py"))


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


a = open(A).read()
if "REASON_PATCH" not in a:
    a = sub(a, "import random, torch, hashlib   # BIND_PATCH\n", "import random, torch, hashlib, copy, collections   # BIND_PATCH REASON_PATCH\n")
    a = sub(a, '"lm_off", "idle", "clause", "flab")', '"lm_off", "idle", "clause", "flab", "carry", "ctarget", "clab", "dlab", "hop")')
    a = sub(a, "        self.toks, self.ev, self.pos, self.env, self.idle, self.lm = [], {}, 0, None, False, False\n",
            "        self.toks, self.ev, self.pos, self.env, self.idle, self.lm = [], {}, 0, None, False, False\n"
            "        self.carry, self.ctarget, self.clab, self.dlab, self.hop = [], [], None, 0, 1\n")
    a = sub(a, '''class Runner:
    """attn=None''', '''def _ptr_bits(ws, tgt):
    labels = [0] * len(ws)
    for i in tgt:
        j = i
        while j < len(ws) - 1 and ws[j] not in PUNCT and j - i < MAXPTR - 1:
            j += 1
        labels[j] |= 1 << (j - i)
    return labels


def chain_info(ws, args, maxh):
    """bridge chain from the answer's clause over clauses linked by shared rare words (count == 2).
    Returns (pointer labels on the link words of the deepest clauses reached, depth)."""
    pos = arg_positions(ws, args) if args else None
    if not pos:
        return None, 0
    ce, cs = clause_ends(ws), clause_starts(ws)
    cnt = collections.Counter(ws)
    argset = set(args)
    depth = {cs[p]: 0 for p in pos}
    link, frontier = {}, set(depth)
    for d in range(1, maxh + 1):
        nxt = {}
        for c in frontier:
            for i in range(c, ce[c] + 1):
                v = ws[i]
                if v in PUNCT or v in argset or cnt[v] != 2:
                    continue
                for j, w in enumerate(ws):
                    if w == v and cs[j] not in depth and cs[j] not in nxt:
                        nxt[cs[j]] = j
        if not nxt:
            break
        for c, j in nxt.items():
            depth[c], link[c] = d, j
        frontier = set(nxt)
    D = max(depth.values())
    if D == 0:
        return None, 0
    return _ptr_bits(ws, [link[c] for c, dd in depth.items() if dd == D]), D


def carry_targets(env, ws_now, maxsteps=4):
    """words of the current observation that a future decision's answer clause needs as a bridge but that will
    not be visible elsewhere in that future observation (oracle roll-out on a copy of the environment)."""
    try:
        e = copy.deepcopy(env)
        a, args = e.oracle()
        obs = e.act(a, args)
    except Exception:
        return []
    now, tg, k = set(ws_now), [], 0
    while not e.done and k < maxsteps:
        a2, args2 = e.oracle()
        O = words(obs)
        pos = arg_positions(O, args2) if args2 else None
        if pos:
            ce, cs = clause_ends(O), clause_starts(O)
            cnt = collections.Counter(O)
            for p in pos:
                for i in range(cs[p], ce[p] + 1):
                    u = O[i]
                    if u not in PUNCT and u not in args2 and cnt[u] == 1 and u in now and u not in tg:
                        tg.append(u)
        obs = e.act(a2, args2)
        k += 1
    return tg


class Runner:
    """attn=None''')
    a = sub(a, '''        self.nch = len(ACTIONS) + (1 if self.bind else 0)          # + focus channel
''', '''        self.hops = int((attn or {}).get("hops") or 1) if self.bind else 1
        self.carry = bool((attn or {}).get("carry"))
        self.cch = len(ACTIONS) + (1 if self.bind else 0)            # carry channel index
        self.nch = len(ACTIONS) + (1 if self.bind else 0) + (1 if self.carry else 0)   # + focus, carry channels
        self._mode = "agent"
''')
    a = sub(a, '''        s.toks, s.ev, s.pos = [SEP, PAD, PAD], {}, 0
        self._push_obs(s, s.env.first)''', '''        s.toks, s.ev, s.pos = [SEP, PAD, PAD], {}, 0
        s.carry, s.hop = [], 1
        self._push_obs(s, s.env.first)''')
    a = sub(a, '''    def _push_obs(self, s, obs):
        s.obs_words = words(obs)
        s.oracle = s.env.oracle() if not s.env.done else None
        s.labels = pointer_labels(s.obs_words, s.oracle[1], bool(self.res.rules.get("ptr_all"))) if s.oracle else None
        s.flab = focus_labels(s.obs_words, s.oracle[1]) if (self.bind and s.oracle) else None
''', '''    def _push_obs(self, s, obs):
        if self.carry and s.carry:                       # self-written note carried from the previous observation
            obs = "note : " + " ".join(s.carry) + " . " + obs
        s.carry = []
        s.obs_words = words(obs)
        s.oracle = s.env.oracle() if not s.env.done else None
        s.labels = pointer_labels(s.obs_words, s.oracle[1], bool(self.res.rules.get("ptr_all"))) if s.oracle else None
        s.dlab = 0
        if self.bind and s.oracle and self.hops >= 2:
            s.flab, s.dlab = chain_info(s.obs_words, s.oracle[1], self.hops)
        else:
            s.flab = focus_labels(s.obs_words, s.oracle[1]) if (self.bind and s.oracle) else None
        s.clab, s.ctarget = None, []
        if self.carry and s.oracle and self._mode != "agent":
            s.ctarget = carry_targets(s.env, s.obs_words)
            if s.ctarget:
                s.clab = _ptr_bits(s.obs_words, [max(i for i, w in enumerate(s.obs_words) if w == u) for u in s.ctarget])
''')
    a = sub(a, '''        rng = random.Random(seed)
        specs = list(reversed(specs))''', '''        rng = random.Random(seed)
        self._mode = mode
        specs = list(reversed(specs))''')
    a = sub(a, '''                            fl = streams[b].flab
                            if fl is not None and wi < len(fl):
                                for k in range(MAXPTR):
                                    if (fl[wi] >> k) & 1:
                                        ri.append(row); ci.append(len(ACTIONS) * MAXPTR + k)
''', '''                            fl = streams[b].flab
                            if fl is not None and wi < len(fl):
                                for k in range(MAXPTR):
                                    if (fl[wi] >> k) & 1:
                                        ri.append(row); ci.append(len(ACTIONS) * MAXPTR + k)
                            cl_ = streams[b].clab
                            if cl_ is not None and wi < len(cl_):
                                for k in range(MAXPTR):
                                    if (cl_[wi] >> k) & 1:
                                        ri.append(row); ci.append(self.cch * MAXPTR + k)
''')
    a = sub(a, '''                        Ys.append(T)
                    collect["dec"].add(Fd, torch.cat(Ys, 1))''', '''                        Ys.append(T)
                    if self.hops >= 2:                      # chain-depth head (0 = no binding, 1, 2)
                        Ys.append(Fn.one_hot(torch.tensor([min(2, streams[b].dlab) for b in drows], device=dev), 3).float())
                    collect["dec"].add(Fd, torch.cat(Ys, 1))''')
    a = sub(a, '''                Q = self._head("q", Fd) if (pred is not None and self.attn is not None) else None
''', '''                Q = self._head("q", Fd) if (pred is not None and self.attn is not None) else None
                if pred is not None and "hop" in self.heads:
                    for r_, h_ in enumerate(self._head("hop", Fd).argmax(1).tolist()):
                        streams[drows[r_]].hop = h_
''')
    a = sub(a, '''                    obs = s.env.act(a, args)
                    if s.env.done:''', '''                    if self.carry:                           # what to write into the note for the next observation
                        if use_oracle:
                            s.carry = list(s.ctarget)
                        else:
                            cs_ = sorted(((float(v[self.cch]), w) for w, v in s.sel.items() if len(v) > self.cch), reverse=True)
                            s.carry = []
                            for v_, w_ in cs_[:2]:
                                if v_ > CARRY_T and w_ < len(s.obs_words) and s.obs_words[w_] not in PUNCT and s.obs_words[w_] not in s.carry:
                                    s.carry.append(s.obs_words[w_])
                    obs = s.env.act(a, args)
                    if s.env.done:''')
    a = sub(a, "MAXW = 192          # max words per observation kept for attention\n",
            "MAXW = 192          # max words per observation kept for attention\nCARRY_T = 0.1       # REASON_PATCH: carry-pointer threshold\n")
    a = sub(a, '''        q = torch.from_numpy(p).float() @ S
        return conf * (ctx @ q).numpy()''', '''        q = torch.from_numpy(p).float() @ S
        h = s.hop if self.hops >= 2 else 1
        if h == 0:
            return np.zeros(n)
        if h == 1:
            return conf * (ctx @ q).numpy()
        # second hop: clauses holding the focus word -> their other rare words form the next query
        m = (S @ q).numpy()
        m[p > 0.2] = 0.0
        m = np.clip(2.0 * (m - 0.5), 0.0, 1.0)
        cid = cs.numpy()
        cw = collections.defaultdict(float)
        for i in range(n):
            cw[cid[i]] = max(cw[cid[i]], m[i])
        cnt = collections.Counter(ws)
        wv = np.array([cw[cid[i]] * (1.0 - m[i]) if (cnt[ws[i]] == 2 and ws[i] not in PUNCT) else 0.0 for i in range(n)])
        if not wv.any():
            return np.zeros(n)
        q2 = torch.from_numpy(wv).float() @ S
        sc = (ctx @ q2).numpy()
        visited = np.array([cw[cid[i]] > 0 or p[i] > 0.2 for i in range(n)])
        fcl = {cid[i] for i in range(n) if p[i] > 0.2}
        visited |= np.array([cid[i] in fcl for i in range(n)])
        sc[visited] = 0.0
        return conf * sc''')
    open(A, "w").write(a)

e = open(Ev).read()
if "REASON_PATCH" not in e:
    e = sub(e, '''"bind": bind, "key_id": res.rules.get("key_id", 0), "key_mix": res.rules.get("key_mix")}''',
            '''"bind": bind, "key_id": res.rules.get("key_id", 0), "key_mix": res.rules.get("key_mix"),
            "hops": res.rules.get("hops", 1), "carry": res.rules.get("carry", 0)}     # REASON_PATCH


def hop_on(cfg):
    return bool(cfg and any(v > 0 for v in (cfg.get("bind") or {}).values()) and (cfg.get("hops") or 1) >= 2)''')
    e = sub(e, '''    return MAXPTR * (len(ACTIONS) + (1 if cfg and any(v > 0 for v in (cfg.get("bind") or {}).values()) else 0))''',
            '''    b = 1 if cfg and any(v > 0 for v in (cfg.get("bind") or {}).values()) else 0
    return MAXPTR * (len(ACTIONS) + b + (1 if cfg and cfg.get("carry") else 0))''')
    e = sub(e, '''        grp = [slice(0, NP), slice(NP, acc[name].C)] if (name == "ptr" and acc[name].C > NP) else None''',
            '''        grp = ([slice(0, NP)] + [slice(c, c + MAXPTR) for c in range(NP, acc[name].C, MAXPTR)]) if (name == "ptr" and acc[name].C > NP) else None''')
    e = sub(e, '''        W = acc[name].W.clone()
''', '''        W = acc[name].W.clone()
        nh = getattr(acc[name], "nhop", 0)
        if nh:                                               # chain-depth head sits after the query columns
            heads["hop"] = W[:, W.shape[1] - nh:].contiguous()
            W = W[:, : W.shape[1] - nh]
''')
    e = sub(e, '''           "dec": Ridge(D, len(ACTIONS) + 4 + (key_dim(cfg) if cfg else 0))}''',
            '''           "dec": Ridge(D, len(ACTIONS) + 4 + (key_dim(cfg) if cfg else 0) + (3 if hop_on(cfg) else 0))}
    if hop_on(cfg):
        acc["dec"].nhop = 3''')
    open(Ev, "w").write(e)

m = open(Mu).read()
if "REASON_PATCH" not in m:
    m = sub(m, '''    if k == "gate":                                           # GATE_PATCH''', '''    if k in ("hops", "carry"):                                # REASON_PATCH
        cur = int(r.rules.get(k) or (1 if k == "hops" else 0))
        new = int(w) if spec else ({1: 2, 2: 1} if k == "hops" else {0: 1, 1: 0})[cur]
        r.rules[k] = new
        return f"readout {'binding depth' if k == 'hops' else 'carried note'} -> {new}"
    if k == "gate":                                           # GATE_PATCH''')
    m = sub(m, '''        k = rng.choice(["bind", "bind", "key_id", "key_mix", "key_mix", "ptr_all", "gate"])''',
            '''        k = rng.choice(["bind", "bind", "key_id", "key_mix", "key_mix", "ptr_all", "gate", "hops", "carry"])''')
    m = sub(m, '''                spec = ("gate", None, None) if a == "gate" else''',
            '''                spec = (a, None, float(w)) if a in ("hops", "carry") else ("gate", None, None) if a == "gate" else''')
    open(Mu, "w").write(m)
print("reason patch applied to", root)
