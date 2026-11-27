"""Closed-loop runner: B parallel streams fed one character per tick through the reservoir.

Readout heads (all linear on the int8 features, none feeding back into the reservoir):
  lm  : next-character prediction ('inner speech' / language perception)
  ptr : at every word end, per action channel, multi-label 'select the word k back' (k < 8) -> argument
        copying; the chosen action's channel decides which words are copied (no cross-task interference)
  act : at the end of the think ticks after ACT, which action to take
  nargs: how many argument words that action takes (the top-n pointed words are copied)

modes: 'oracle' (teacher forcing), 'agent' (evaluation), 'dagger' (agent acts, oracle labels;
each decision is handed to the oracle with probability beta).
"""
import random, torch, hashlib, copy, collections, os   # BIND_PATCH REASON_PATCH GATE2_PATCH
import numpy as np
import torch.nn.functional as Fn
from .reservoir import encode, PAD, ACT, SEP, VOCAB, C2I
from .env import FAMILIES, ACTIONS, A2I, MAXPTR, PUNCT, words, pointer_labels

SPACE = C2I[" "]


class Stream:
    __slots__ = ("toks", "ev", "pos", "env", "obs_words", "labels", "sel", "oracle", "spec", "lm", "lm_off", "idle", "clause", "flab", "carry", "ctarget", "clab", "dlab", "hop", "note", "ncar")   # CARRY2_PATCH

    def __init__(self):
        self.toks, self.ev, self.pos, self.env, self.idle, self.lm = [], {}, 0, None, False, False
        self.carry, self.ctarget, self.clab, self.dlab, self.hop, self.note, self.ncar = [], [], None, 0, 1, [], 0

    def push_text(self, text, word_events=True):
        base = len(self.toks)
        ids = encode(text + " ")
        wi = -1
        for i, t in enumerate(ids):
            if t != SPACE and (i == 0 or ids[i - 1] == SPACE):
                wi += 1
            if t == SPACE and i > 0 and ids[i - 1] != SPACE and word_events:
                self.ev[base + i] = ("w", wi)
        self.toks += ids


def clause_ends(ws):
    """index of the clause end (next punctuation word, else last word) for every word."""
    out, j = [0] * len(ws), len(ws) - 1
    for i in range(len(ws) - 1, -1, -1):
        if ws[i] in PUNCT:
            j = i
        out[i] = j
    return out


def arg_positions(ws, args):
    """word positions the oracle's arguments are copied from (same choice rule as pointer_labels)."""
    pos, last = [], -1
    for a in args:
        cand = [i for i, w in enumerate(ws) if w == a and i > last] or [i for i, w in enumerate(ws) if w == a]
        if not cand:
            return None
        i = cand[-1] if len(args) == 1 else cand[0]
        pos.append(i); last = i
    return pos


MAXW = 192          # max words per observation kept for attention
CARRY_T = 0.1       # REASON_PATCH: carry-pointer threshold
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

    def solve(self, alpha=1e-2, w0=None):
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
        w = torch.linalg.solve(Cn, b) / sd
        return w.float().reshape(NCOMP, self.ds)
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


def _ptr_bits(ws, tgt):
    labels = [0] * len(ws)
    for i in tgt:
        j = i
        while j < len(ws) - 1 and ws[j] not in PUNCT and j - i < MAXPTR - 1:
            j += 1
        labels[j] |= 1 << (j - i)
    return labels


def rare_fn(ws):
    """CARRY3_PATCH: word -> is a link word (twice in the text, or once in the text + in the note)."""
    k = ws.index(".") + 1 if len(ws) > 2 and ws[0] == "note" and ws[1] == ":" and "." in ws else 0
    noted = set(ws[2:k - 1]) if k else set()
    cnt = collections.Counter(ws[k:])
    return lambda v: v not in PUNCT and (cnt[v] == 2 or (cnt[v] == 1 and v in noted))


def chain_info(ws, args, maxh):
    """bridge chain from the answer's clause over clauses linked by shared rare words (count == 2).
    Returns (pointer labels on the link words of the deepest clauses reached, depth)."""
    pos = arg_positions(ws, args) if args else None
    if not pos:
        return None, 0
    ce, cs = clause_ends(ws), clause_starts(ws)
    rare = rare_fn(ws)
    argset = set(args)
    depth = {cs[p]: 0 for p in pos}
    link, frontier = {}, set(depth)
    for d in range(1, maxh + 1):
        nxt = {}
        for c in frontier:
            for i in range(c, ce[c] + 1):
                v = ws[i]
                if v in argset or not rare(v):
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


def carry_cap(cfg):
    """MEMO2_PATCH: most words one step may write into the note."""
    c2 = int((cfg or {}).get("carry") or 0) >= 2
    nl = int((cfg or {}).get("note_len") or 4)
    return 2 + c2 if nl <= 4 else nl - 1


def carry_targets(env, ws_now, maxsteps=4, args_too=False):
    """words of the current observation that a future decision's answer clause needs as a bridge but that will
    not be visible elsewhere in that future observation (oracle roll-out on a copy of the environment)."""
    try:
        e = copy.deepcopy(env)
        a, args = e.oracle()
        obs = e.act(a, args)
    except Exception:
        return []
    now, tg, k = set(ws_now), [], 0
    atg = []                                          # STAGE5_PATCH: argument targets (args first, then bridge words)
    nxt = set(words(obs))                             # carry an argument only when it is about to leave view
    if args_too and hasattr(env, "memo"):             # MEMO_PATCH: words a future DECISION depends on
        atg += [u for u in env.memo() if u in now and u not in atg]   # carried even if it reappears: it is the comparand
    while not e.done and k < maxsteps:
        a2, args2 = e.oracle()
        O = words(obs)
        pos = arg_positions(O, args2) if args2 else None
        if args_too and args2 and not pos:            # STAGE5_PATCH: a needed argument that will be out of view
            for u in args2:
                if u not in O and u in now and u not in nxt and u not in PUNCT and u not in atg:
                    atg.append(u)
        if pos and (not args_too or a2 == "answer"):
            ce, cs = clause_ends(O), clause_starts(O)
            cnt = collections.Counter(O)
            for p in pos:
                for i in range(cs[p], ce[p] + 1):
                    u = O[i]
                    if u not in PUNCT and u not in args2 and cnt[u] == 1 and u in now and u not in tg:
                        tg.append(u)
        obs = e.act(a2, args2)
        k += 1
    return (atg + [u for u in tg if u not in atg and u != ">" and u not in ACTIONS]) if args_too else tg


class Runner:
    """attn=None: argument words chosen by the per-word pointer scores (unary).
    attn=dict(R=(D,K) fixed random ternary projection, mode='attn'|'sum'): query-key attention at decision
    time -- the 'q' head (fitted jointly with the decision heads) maps the decision state to a query that is
    matched against every word's key = [R f(word end), R f(clause end)], centred over the observation."""
    def __init__(self, res, heads=None, corpus_ids=None, attn=None):
        self.res, self.heads, self.corpus, self.attn = res, heads or {}, corpus_ids, attn
        self.bind = bool(attn and any(v > 0 for v in (attn.get("bind") or {}).values()))
        self.hops = int((attn or {}).get("hops") or 1) if self.bind else 1
        self.carry = bool((attn or {}).get("carry"))
        self.carry2 = int((attn or {}).get("carry") or 0) >= 2           # STAGE5_PATCH: note may carry arguments
        self.ccap = carry_cap(attn)                                       # MEMO2_PATCH
        self.carry3 = int((attn or {}).get("carry") or 0) >= 3           # CARRY4_PATCH: note written in text order
        self.cch = len(ACTIONS) + (1 if self.bind else 0)            # carry channel index
        self.nch = len(ACTIONS) + (1 if self.bind else 0) + (1 if self.carry else 0)   # + focus, carry channels
        self._mode = "agent"

    def _head(self, name, F):
        W = self.heads[name]
        return F.float() @ W[:-1] + W[-1]

    def _new_episode(self, s, specs, mode):
        if not specs:
            s.idle, s.env = True, None
            return
        fam, seed, split = specs.pop()
        s.spec = (fam, seed, split)
        s.env = FAMILIES[fam](random.Random(seed), split)
        s.toks, s.ev, s.pos = [SEP, PAD, PAD], {}, 0
        s.carry, s.hop, s.note, s.ncar = [], 1, [], 0
        self._push_obs(s, s.env.first)

    def _push_obs(self, s, obs):
        if self.carry:                                   # persistent self-written note (max 4 words)
            for w_ in s.carry:
                if w_ not in s.note:
                    s.note.append(w_)
            s.note = s.note[-int(self.res.rules.get("note_len") or 4):]   # MEMO_PATCH: evolvable note length
            if s.note:
                obs = "note : " + " ".join(s.note) + " . " + obs
        s.carry = []
        s.obs_words = words(obs)
        s.oracle = s.env.oracle() if not s.env.done else None
        s.labels = pointer_labels(s.obs_words, s.oracle[1], (2 if (self.res.rules.get("occ") == 3 and s.oracle[0] == "answer") else bool(self.res.rules.get("ptr_all") or self.res.rules.get("occ")))) if s.oracle else None
        s.dlab = 0
        if self.bind and s.oracle and self.hops >= 2:
            s.flab, s.dlab = chain_info(s.obs_words, s.oracle[1], self.hops)
        else:
            s.flab = focus_labels(s.obs_words, s.oracle[1]) if (self.bind and s.oracle) else None
        s.clab, s.ctarget = None, []
        if self.carry and s.oracle and self._mode != "agent":
            s.ctarget = [u for u in (carry_targets(s.env, s.obs_words, 8, True) if self.carry2 else carry_targets(s.env, s.obs_words))
                         if u not in s.note][:self.ccap]   # MEMO2_PATCH
            if self.carry3:                              # CARRY4_PATCH: text order
                s.ctarget.sort(key=lambda u: s.obs_words.index(u) if u in s.obs_words else 10 ** 6)
            if s.ctarget:
                s.clab = _ptr_bits(s.obs_words, [max(i for i, w in enumerate(s.obs_words) if w == u) for u in s.ctarget])
        s.sel = {}
        s.clause = clause_ends(s.obs_words)
        s.push_text(obs)
        if not s.env.done:
            s.toks += [ACT] + [PAD] * self.res.rules["think"]
            s.ev[len(s.toks) - 1] = ("d",)

    @torch.no_grad()
    def run(self, specs, mode="agent", collect=None, beta=0.3, n_lm=0, lm_every=2, B=256, seed=0, max_ticks=10 ** 6):
        rng = random.Random(seed)
        self._mode = mode
        specs = list(reversed(specs))
        n_task = B - n_lm
        streams = [Stream() for _ in range(B)]
        g = torch.Generator().manual_seed(seed)
        for i, s in enumerate(streams):
            if i >= n_task:
                s.lm, s.lm_off = True, int(torch.randint(0, len(self.corpus) - 10 ** 6, (1,), generator=g))
            else:
                self._new_episode(s, specs, mode)
        gs = self.res.graph(B)
        results, tick = [], 0
        dev = self.res.device
        if n_lm:
            if getattr(self, "_cdev", None) is None or self._cdev_src is not self.corpus:
                self._cdev, self._cdev_src = self.corpus.to(dev), self.corpus
            lm_offs = torch.tensor([s.lm_off for s in streams[n_task:]], device=dev)
        NC = MAXPTR * self.nch
        if self.attn is not None:
            R = self.attn["R"]
            K = R.shape[1]
            kbuf = torch.zeros(n_task, MAXW, K, device=dev)
        while tick < max_ticks:
            if all(s.idle for s in streams[:n_task]) and (n_task > 0 or tick >= max_ticks):
                break
            toks = [PAD if (s.idle or s.pos >= len(s.toks)) else s.toks[s.pos] for s in streams[:n_task]]
            tt = torch.tensor(toks, dtype=torch.long).to(dev, non_blocking=True)
            if n_lm:
                tt = torch.cat([tt, self._cdev[lm_offs + tick]])
            Fg = gs(tt)
            F = None
            wrows, wlab, drows = [], [], []
            for b in range(n_task):
                s = streams[b]
                if s.idle or s.pos >= len(s.toks):
                    continue
                e = s.ev.get(s.pos)
                s.pos += 1
                if e is None:
                    continue
                if e[0] == "w":
                    wrows.append((b, e[1]))
                else:
                    drows.append(b)
            if wrows or drows or (n_lm and tick % lm_every == 0):
                F = Fg
            # ---- pointer events ----
            if wrows:
                idx = torch.tensor([b for b, _ in wrows], device=dev)
                Fw = F[idx]
                if self.attn is not None:
                    wi_t = torch.tensor([min(wi, MAXW - 1) for _, wi in wrows], device=dev)
                    kbuf[idx, wi_t] = Fw.float() @ R
                if collect is not None and "ptr" in collect:
                    keep, ri, ci = [], [], []
                    for r, (b, wi) in enumerate(wrows):
                        lab = streams[b].labels
                        if lab is not None and wi < len(lab):
                            c = A2I[streams[b].oracle[0]] * MAXPTR             # channel of the oracle's action
                            row = len(keep)
                            keep.append(r)
                            for k in range(MAXPTR):
                                if (lab[wi] >> k) & 1:
                                    ri.append(row); ci.append(c + k)
                            fl = streams[b].flab
                            if fl is not None and wi < len(fl):
                                for k in range(MAXPTR):
                                    if (fl[wi] >> k) & 1:
                                        ri.append(row); ci.append(len(ACTIONS) * MAXPTR + k)
                            cl_ = streams[b].clab
                            if cl_ is not None and wi < len(cl_):
                                for k in range(MAXPTR):
                                    if (cl_[wi] >> k) & 1:
                                        ri.append(row); ci.append(self.cch * MAXPTR + k)
                    if keep:
                        Y = torch.zeros(len(keep), NC)
                        if ri:
                            Y[ri, ci] = 1.0
                        collect["ptr"].add(Fw[torch.tensor(keep).to(dev)], Y.to(dev))
                if mode != "oracle" and "ptr" in self.heads:
                    P = self._head("ptr", Fw).view(len(wrows), self.nch, MAXPTR).cpu().numpy()
                    for (b, wi), sc in zip(wrows, P):
                        d = streams[b].sel
                        for k in range(min(MAXPTR, wi + 1)):
                            w = wi - k
                            d[w] = np.maximum(d[w], sc[:, k]) if w in d else sc[:, k].copy()
            # ---- decisions ----
            if drows:
                idx = torch.tensor(drows, device=dev)
                Fd = F[idx]
                if self.attn is not None:
                    keys = [self._keys(kbuf, b, streams[b]) for b in drows]
                if collect is not None and "dec" in collect:     # one Gram for all decision heads (act | nargs | query)
                    Ya = Fn.one_hot(torch.tensor([A2I[streams[b].oracle[0]] for b in drows], device=dev), len(ACTIONS)).float()
                    Yn = Fn.one_hot(torch.tensor([min(3, len(streams[b].oracle[1])) for b in drows], device=dev), 4).float()
                    Ys = [Ya, Yn]
                    if self.attn is not None:              # query target: mean (centred) key of the oracle's argument words
                        T = torch.zeros(len(drows), keys[0].shape[1] if keys else 2 * K, device=dev)
                        for r, b in enumerate(drows):
                            ps = arg_positions(streams[b].obs_words, streams[b].oracle[1])
                            if ps:
                                T[r] = keys[r][[min(p, MAXW - 1) for p in ps]].mean(0)
                        Ys.append(T)
                    if self.hops >= 2:                      # chain-depth head (0 = no binding, 1, 2)
                        Ys.append(Fn.one_hot(torch.tensor([min(2, streams[b].dlab) for b in drows], device=dev), 3).float())
                    if self.carry:                          # how many words to write into the note (0, 1, 2)
                        Ys.append(Fn.one_hot(torch.tensor([min(self.ccap, len(streams[b].ctarget)) for b in drows], device=dev), self.ccap + 1).float())
                    collect["dec"].add(Fd, torch.cat(Ys, 1))
                pred = self._head("act", Fd).argmax(1).tolist() if (mode != "oracle" and "act" in self.heads) else None
                nargs = self._head("nargs", Fd).argmax(1).tolist() if pred is not None else None
                Q = self._head("q", Fd) if (pred is not None and self.attn is not None) else None
                if pred is not None and "ncarry" in self.heads:
                    for r_, n_ in enumerate(self._head("ncarry", Fd).argmax(1).tolist()):
                        streams[drows[r_]].ncar = n_
                if pred is not None and "hop" in self.heads:
                    for r_, h_ in enumerate(self._head("hop", Fd).argmax(1).tolist()):
                        streams[drows[r_]].hop = h_
                gate_on = self.attn is not None and ("gate" in self.heads or (collect is not None and "gate" in collect))
                if gate_on:
                    Sd = Fd.float() @ self.attn["R"]
                    Sd = Sd / (Sd.norm(dim=1, keepdim=True) + 1e-6)
                for r, b in enumerate(drows):
                    s = streams[b]
                    use_oracle = mode in ("oracle", "gate") or pred is None or (mode == "dagger" and rng.random() < beta)
                    if use_oracle:
                        a, args = s.oracle
                        if mode == "gate" and gate_on and a != "done" and (self._mix(a) > 0 or self._bindw(a) > 0):
                            Z = self._components(keys[r], Q[r], s, A2I[a])
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
                                    y = [0.0]
                            if any(y):
                                collect["gate"].add(Z, self._gstate(Sd[r], A2I[a]), y)
                    else:
                        a = ACTIONS[pred[r]]
                        # copy the n best-pointed words of the observation (n from the nargs head), in text order
                        ai = A2I[a]                                  # the chosen action's pointer channel
                        if self.attn is not None and a != "done" and (self._mix(a) > 0 or self._bindw(a) > 0):
                            sc = self._gated(keys[r], Q[r], s, ai, Sd[r]) if "gate" in self.heads else self._attn_scores(keys[r], Q[r], s, ai)
                            top = sorted(range(len(sc)), key=lambda w: sc[w], reverse=True)
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
                        top = top[: nargs[r]]
                        args = [s.obs_words[i] for i in sorted(top) if i < len(s.obs_words)]
                    if self.carry:                           # what to write into the note for the next observation
                        if use_oracle:
                            s.carry = list(s.ctarget)
                        else:
                            cs_ = sorted(((float(v[self.cch]), w) for w, v in s.sel.items() if len(v) > self.cch), reverse=True)
                            s.carry = []
                            for v_, w_ in cs_:
                                if len(s.carry) >= s.ncar:
                                    break
                                if w_ < len(s.obs_words) and s.obs_words[w_] not in PUNCT and s.obs_words[w_] not in s.carry \
                                        and s.obs_words[w_] not in s.note:
                                    s.carry.append(s.obs_words[w_])
                            if self.carry3:                  # CARRY4_PATCH
                                s.carry.sort(key=lambda u: s.obs_words.index(u))
                    obs = s.env.act(a, args)
                    if s.env.done:
                        results.append({"family": s.spec[0], "seed": s.spec[1], "success": bool(s.env.success), "steps": s.env.steps})
                        self._new_episode(s, specs, mode)
                    else:
                        s.toks, s.ev, s.pos = [], {}, 0
                        self._push_obs(s, obs)
            # ---- language model rows ----
            if n_lm and tick % lm_every == 0 and collect is not None and "lm" in collect:
                nxt = self._cdev[lm_offs + tick + 1]
                collect["lm"].add(F[n_task:], Fn.one_hot(nxt, VOCAB).float())
            tick += 1
        return results, tick

    def _keys(self, kbuf, b, s):
        """(n_words, 2K) keys of the current observation: own word-end projection | its clause-end projection, centred."""
        n = min(len(s.obs_words), MAXW)
        own = kbuf[b, :n]
        cl = kbuf[b, torch.tensor([min(c, MAXW - 1) for c in s.clause[:n]], device=kbuf.device)]
        k = torch.cat([own, cl], 1)
        kid = self.attn.get("key_id", 0) or 0
        if kid > 0:                                           # + the word's own identity signature
            S = torch.stack([sig(w) for w in s.obs_words[:n]]).to(k.device)
            S = S - S.mean(0, keepdim=True)
            k = k - k.mean(0, keepdim=True)
            scale = kid * k.norm(dim=1).mean() / (S.norm(dim=1).mean() + 1e-6)
            return torch.cat([k, scale * S], 1)
        return k - k.mean(0, keepdim=True)

    def _gstate(self, sd, ai):
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
        if int(self.res.rules.get("gate") or 0) == 2:            # GATE2_PATCH: masked, occ-pooled components
            Z = self._components2(k, q, s, ai)
            g = (self.heads["gate"] @ self._gstate(sd, ai)).cpu().numpy()
            sc = Z @ g
            ok = np.array([w in s.sel for w in range(len(sc))])
            return list(np.where(ok, sc, -1e9)) if ok.any() else list(sc)
        Z = self._components(k, q, s, ai)
        g = (self.heads["gate"] @ self._gstate(sd, ai)).cpu().numpy()          # (NCOMP,)
        return list(Z @ g)

    def _components2(self, k, q, s, ai):
        """GATE2_PATCH: like _components, but the unary score is occurrence-pooled exactly as in _attn_scores."""
        Z = self._components(k, q, s, ai)
        occ = int(self.res.rules.get("occ") or 0)
        if occ and ACTIONS[ai] == "answer" and s.sel:
            n = Z.shape[0]
            u = np.array([float(s.sel[w][ai]) if w in s.sel else -1e9 for w in range(n)])
            u = self._occ_pool_rec(u, s.obs_words)           # OCCREC_PATCH
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

    def _bindw(self, a):
        return (self.attn.get("bind") or {}).get(a, 0.0) if self.attn is not None else 0.0

    def _bind_scores(self, s, n):
        """two-hop bridge attention: focus-pointer-weighted identity query . clause-context signatures."""
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
        if self.res.rules.get("bmask"):                 # BMASK_PATCH: a bridge word must occur elsewhere too
            cnt_ = collections.Counter(ws)
            f = np.where(np.array([cnt_[w] >= 2 for w in ws]), f, -1e9)
        ok = f > -1e8
        if not ok.any() or f[ok].max() < 0.06:         # no confident bridge word: no binding term
            return np.zeros(n)
        conf = min(1.0, float(f[ok].max()) / 0.15)      # pointer outputs are small (sparse labels): scale, don't cut
        z = np.where(ok, (f - f[ok].mean()) / (f[ok].std() + 1e-6), -1e9)
        p = np.exp(4.0 * (z - z.max()))
        p /= p.sum()
        q = torch.from_numpy(p).float() @ S
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
        rare = rare_fn(ws)
        wv = np.array([cw[cid[i]] * (1.0 - m[i]) if rare(ws[i]) else 0.0 for i in range(n)])
        if not wv.any():
            return np.zeros(n)
        q2 = torch.from_numpy(wv).float() @ S
        sc = (ctx @ q2).numpy()
        visited = np.array([cw[cid[i]] > 0 or p[i] > 0.2 for i in range(n)])
        fcl = {cid[i] for i in range(n) if p[i] > 0.2}
        visited |= np.array([cid[i] in fcl for i in range(n)])
        sc[visited] = 0.0
        return conf * sc

    def _mix(self, a):
        """weight of attention vs the unary pointer for action a (0 = unary only)."""
        mix = self.attn.get("mix")
        return mix.get(a, 0.0) if mix is not None else 1.0

    def _occ_pool_rec(self, u, ws):
        """OCCREC_PATCH: z-score the candidate scores, add beta * position / n, then pool over occurrences."""
        beta = float(self.res.rules.get("occ_rec") or 0.0)
        u = np.asarray(u, dtype=float)
        ok = u > -1e8
        if beta == 0.0 or not ok.any():
            return self._occ_pool(u, ws)
        z = np.full(len(u), -1e9)
        z[ok] = (u[ok] - u[ok].mean()) / (u[ok].std() + 1e-6) + beta * np.arange(len(u))[ok] / max(1, len(u))
        return self._occ_pool(z, ws)

    @staticmethod
    def _occ_pool(v, ws):
        """OCC_PATCH: every position takes the max score over all occurrences of its word."""
        best = {}
        for i, x in enumerate(v):
            w = ws[i] if i < len(ws) else None
            if w not in best or x > best[w]:
                best[w] = x
        return np.array([best[ws[i] if i < len(ws) else None] for i in range(len(v))])

    def _attn_scores(self, k, q, s, ai):
        K2 = 2 * self.attn["R"].shape[1]
        if (self.attn.get("key_id", 0) or 0) > 0 and k.shape[1] > K2:     # identity part weighted per action
            wm = (self.attn.get("key_mix") or {"answer": 1.0}).get(ACTIONS[ai], 0.0)
            a = (k[:, :K2] @ q[:K2] + wm * (k[:, K2:] @ q[K2:])).cpu().numpy()
        else:
            a = (k @ q).cpu().numpy()
        lam = self._mix(ACTIONS[ai])
        if (self.attn.get("mode", "attn") == "sum" or self.attn.get("mix") is not None) and s.sel:
            u = np.array([float(s.sel[w][ai]) if w in s.sel else -1e9 for w in range(len(a))])
            occ = int(self.res.rules.get("occ") or 0)
            if occ and ACTIONS[ai] == "answer":
                u = self._occ_pool_rec(u, s.obs_words)       # OCCREC_PATCH
                if occ == 2:                                  # OCC3_PATCH
                    a = self._occ_pool(np.asarray(a), s.obs_words)
            ok = u > -1e8
            a = (a - a.mean()) / (a.std() + 1e-6)
            u2 = np.where(ok, (u - u[ok].mean()) / (u[ok].std() + 1e-6), -1e3)
            a = lam * a + u2
        bw = self._bindw(ACTIONS[ai])
        if bw > 0:
            bs = self._bind_scores(s, len(a))
            a = np.asarray(a) + 3.0 * bw * (bs - np.median(bs))     # identity match ~1 x confidence, noise ~0.1
        return list(a)

    @torch.no_grad()
    def lm_eval(self, text_ids, B=64, T=400, warm=50, seed=123):
        """next-character accuracy on held-out text (same positions every time)."""
        g = torch.Generator().manual_seed(seed)
        offs = torch.randint(0, len(text_ids) - T - 2, (B,), generator=g)
        seq = torch.stack([text_ids[o:o + T + 1] for o in offs]).to(self.res.device)
        gs = self.res.graph(B)
        per = torch.zeros(B, device=self.res.device)
        for t in range(T):
            F = gs(seq[:, t])
            if t >= warm:
                ok = (self._head("lm", F).argmax(1) == seq[:, t + 1]).float()
                per += ok
        per /= (T - warm)
        return per.mean().item(), per.cpu().tolist()
