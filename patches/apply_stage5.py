"""Patch: curriculum stage 5 -- extended agentic tasks without scaffolding (STAGE5_PATCH). Idempotent.

Stage 4 is at final 0.963. Every agentic family so far restates the task or hints the next step ('next look up X',
'now install Y', 'compute it', 'save it as K'). Stage 5 removes the hand-holding:
  multi      two subgoals in one task: install a tool, THEN open a file on a site and report a field. After the
             install nothing restates the second subgoal -> the file / site names must be kept (note).
  retry      flaky tools: a call times out ('error : timeout .') up to twice; the agent must repeat the same call
             (arguments recoverable from the action echo) and then answer.
  research2  the research task (look up 2 populations, remember both, recall, add, save as K) with NO hints:
             tool replies are bare ('saved .', 'result : 42 .'); the agent must plan the 7-step sequence and keep
             the second country and the key in its note.
Mechanism (agent): carry >= 2 -- the note may also carry a future action's ARGUMENT that will be missing from
the future observation (oracle roll-out up to 8 steps); before, only bridge words of answer clauses were carried,
so an argument that disappeared from view could never be produced.
Usage: python apply_stage5.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
Ev, A, Mu = (os.path.join(root, "trn", f) for f in ("env.py", "agent.py", "mutate.py"))


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


NEW = '''
# ---------------- stage 5: extended agentic tasks without scaffolding (STAGE5_PATCH) ----------------
class Multi(Env):
    """install a tool, then open a file on a site and report a field; the second subgoal is never restated."""
    family, stage, max_steps = "multi", 5, 9

    def __init__(self, rng, split):
        super().__init__(rng, split)
        self.pk = packages(rng, split, rng.randint(2, 3))
        self.target = rng.choice(self.pk)
        self.site, self.file = uniq_names(rng, split, 2, (2, 2), avoid=tuple(n for n, _, _ in self.pk))
        self.owner, self.size = rng.choice(PEOPLE), str(rng.randint(2, 900))
        self.key = rng.choice(["owner", "size"])
        self.first = (f"task : install the tool that {self.target[1]} {self.target[2]} files , then open the file "
                      f"{self.file} on {self.site} and tell me its {self.key} .")
        self.answer = [self.owner if self.key == "owner" else self.size]
        self.searched, self.installed, self.opened = False, False, False

    def tool(self, a, args):
        if a == "search":
            if args == [self.target[1], self.target[2]]:
                self.searched = True
                return "results : " + " ".join(f"{n} {v} {f} files ;" for n, v, f in self.pk)
            return "results : none ."
        if a == "install" and len(args) == 1:
            if args[0] == self.target[0]:
                self.installed = True
                return f"installed {args[0]} ."
            return "error : wrong tool ."
        if a == "open" and self.installed:
            if sorted(args) == sorted([self.site, self.file]):
                self.opened = True
                return f"file {self.file} : owner {self.owner} ; size {self.size} kb ."
            return "error : not found ."
        if a == "open":
            return "error : install the tool first ."
        return "error : nothing happens ."

    def oracle(self):
        if self.opened:
            return "answer", self.answer
        if self.installed:
            return "open", [self.file, self.site]
        if self.searched:
            return "install", [self.target[0]]
        return "search", [self.target[1], self.target[2]]


class Retry(Env):
    """flaky tools: lookups time out up to twice; repeat the call, then answer."""
    family, stage, max_steps = "retry", 5, 5

    def __init__(self, rng, split):
        super().__init__(rng, split)
        self.c, f = country_facts(rng, split)
        self.facts = f
        self.key = rng.choice(["capital", "population", "language"])
        self.fails = rng.choice([0, 1, 1, 2])
        self.first = f"task : look up the {self.key} of {self.c} . the service is unreliable ."
        self.answer = [f[self.key]]
        self.got = False

    def tool(self, a, args):
        if a == "lookup" and args == [self.c]:
            if self.fails > 0:
                self.fails -= 1
                return "error : timeout ."
            self.got = True
            f = self.facts
            return f"{self.c} : capital {f['capital']} ; population {f['population']} million ; language {f['language']} ."
        if a == "lookup":
            return "error : not in table ."
        return "error : nothing happens ."

    def oracle(self):
        return ("answer", self.answer) if self.got else ("lookup", [self.c])


class Research2(Env):
    """research without hints: lookup x2, remember x2, recall, calc, save as key -- bare tool replies."""
    family, stage, max_steps = "research2", 5, 11

    def __init__(self, rng, split):
        super().__init__(rng, split)
        (self.c1, f1), (self.c2, f2) = country_facts(rng, split), country_facts(rng, split)
        while self.c2 == self.c1:
            self.c2, f2 = country_facts(rng, split)
        self.p = {self.c1: f1["population"], self.c2: f2["population"]}
        self.key = make_name(rng, split, (2, 2))
        self.first = f"task : add the population of {self.c1} and {self.c2} and save the total as {self.key} ."
        self.total = str(int(self.p[self.c1]) + int(self.p[self.c2]))
        self.mem, self.saved, self.last = {}, False, ""

    def tool(self, a, args):
        if a == "lookup" and len(args) == 1 and args[0] in self.p:
            return f"{args[0]} : population {self.p[args[0]]} million ."
        if a == "remember" and len(args) == 2:
            if sorted(args) == sorted([self.key, self.total]):
                self.saved = True
                return "saved ."
            self.mem[args[0]] = args[1]
            return "saved ."
        if a == "recall" and len(args) == 2 and all(x in self.mem for x in args):
            return f"memory : {args[0]} {self.mem[args[0]]} plus {args[1]} {self.mem[args[1]]} ."
        if a == "calc":
            r = calculator(args)
            return "error : bad expression ." if r is None else f"result : {r} ."
        return "error : that did not work ."

    def goal_reached(self):
        return self.saved

    def act(self, a, args):
        out = super().act(a, args)
        self.last = out
        return out

    def oracle(self):
        if self.saved:
            return "done", []
        ok = {c: self.mem.get(c) == self.p[c] for c in (self.c1, self.c2)}
        last = self.last
        if ok[self.c1] and ok[self.c2]:
            if "result :" in last:
                return "remember", [self.key, self.total]
            if "memory :" in last:
                return "calc", [self.p[self.c1], "plus", self.p[self.c2]]
            return "recall", [self.c1, self.c2]
        c = self.c1 if not ok[self.c1] else self.c2
        if f"{c} : population {self.p[c]}" in last:
            return "remember", [c, self.p[c]]
        return "lookup", [c]
'''

e = open(Ev).read()
if "STAGE5_PATCH" not in e:
    e = sub(e, "                                     Hop2, Chain, ToolPick2, Instr2, WebDeep)}",
            "                                     Hop2, Chain, ToolPick2, Instr2, WebDeep, Multi, Retry, Research2)}")
    e = sub(e, "\n\nFAMILIES = {c.family: c for c in (Facts,", NEW + "\n\nFAMILIES = {c.family: c for c in (Facts,")
    e = sub(e, "for s in range(5)}", "for s in range(6)}")
    e = sub(e, '''"multi-hop reasoning, carried goals, compositional generalisation"]''',
            '''"multi-hop reasoning, carried goals, compositional generalisation",
               "extended agentic tasks without scaffolding (subgoals, retries, unhinted plans)"]''')
    open(Ev, "w").write(e)

a = open(A).read()
if "STAGE5_PATCH" not in a:
    a = sub(a, '''def carry_targets(env, ws_now, maxsteps=4):''', '''def carry_targets(env, ws_now, maxsteps=4, args_too=False):''')
    a = sub(a, '''    now, tg, k = set(ws_now), [], 0
''', '''    now, tg, k = set(ws_now), [], 0
    atg = []                                          # STAGE5_PATCH: argument targets (args first, then bridge words)
    nxt = set(words(obs))                             # carry an argument only when it is about to leave view
''')
    a = sub(a, '''        obs = e.act(a2, args2)
        k += 1
    return tg
''', '''        obs = e.act(a2, args2)
        k += 1
    return (atg + [u for u in tg if u not in atg and u != ">" and u not in ACTIONS]) if args_too else tg
''')
    a = sub(a, '''        O = words(obs)
        pos = arg_positions(O, args2) if args2 else None
        if pos:''', '''        O = words(obs)
        pos = arg_positions(O, args2) if args2 else None
        if args_too and args2 and not pos:            # STAGE5_PATCH: a needed argument that will be out of view
            for u in args2:
                if u not in O and u in now and u not in nxt and u not in PUNCT and u not in atg:
                    atg.append(u)
        if pos and (not args_too or a2 == "answer"):''')
    a = sub(a, '''            s.ctarget = [u for u in carry_targets(s.env, s.obs_words) if u not in s.note][:2]''',
            '''            s.ctarget = [u for u in (carry_targets(s.env, s.obs_words, 8, True) if self.carry2 else carry_targets(s.env, s.obs_words))
                         if u not in s.note][:2 + self.carry2]''')
    a = sub(a, '''        self.carry = bool((attn or {}).get("carry"))''', '''        self.carry = bool((attn or {}).get("carry"))
        self.carry2 = int((attn or {}).get("carry") or 0) >= 2           # STAGE5_PATCH: note may carry arguments''')
    a = sub(a, '''                        Ys.append(Fn.one_hot(torch.tensor([min(2, len(streams[b].ctarget)) for b in drows], device=dev), 3).float())''',
            '''                        Ys.append(Fn.one_hot(torch.tensor([min(2 + self.carry2, len(streams[b].ctarget)) for b in drows], device=dev), 3 + self.carry2).float())''')
    open(A, "w").write(a)

v = open(os.path.join(root, "trn", "evolve.py")).read()
if "STAGE5_PATCH" not in v:     # carry >= 2: the note-write count head has 4 classes (0-3 words)
    v = sub(v, '''([("ncarry", 3)] if cfg and cfg.get("carry") else [])''',
            '''([("ncarry", 3 + (int(cfg.get("carry")) >= 2))] if cfg and cfg.get("carry") else [])   # STAGE5_PATCH''')
    v = sub(v, '''+ (3 if cfg and cfg.get("carry") else 0))}''', '''+ ((3 + (int(cfg.get("carry")) >= 2)) if cfg and cfg.get("carry") else 0))}''')
    open(os.path.join(root, "trn", "evolve.py"), "w").write(v)

m = open(Mu).read()
if "STAGE5_PATCH" not in m:
    m = sub(m, '''        new = int(w) if spec else ({1: 2, 2: 1} if k == "hops" else {0: 1, 1: 0})[cur]''',
            '''        new = int(w) if spec else ({1: 2, 2: 1} if k == "hops" else {0: 1, 1: 2, 2: 1})[cur]   # STAGE5_PATCH: carry 2 = args too''')
    open(Mu, "w").write(m)
print("stage5 patch applied to", root)
