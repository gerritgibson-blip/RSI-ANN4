"""Patch: curriculum stage 4 -- harder agentic reasoning and generalisation (stage 3 is solved: 0.966). Idempotent.

New families (all previous families stay active):
  hop2      two-hop facts: 'what is the version of the package that X needs ?' over shuffled 'X needs D .' and
            'D version V .' lines with distractors (one-hop binding lands on D, the wrong answer)
  chain     multi-step with a goal carried across observations: 'look up the capital of X , then look up its
            <key> .' -- the second lookup's result never restates which field was asked for
  toolpick2 compositional generalisation: request = frame x phrase; (frame, phrase) pairs are split into
            train / test, so every test request is a combination never seen in training
  instr2    multiple / novel corrections ('no wait ... actually , go back to x', 'sorry , i meant x',
            'do not v x or y . v z .')
  webdeep   sites up to 3 links deep, 5 sections, 6 items with decoy old prices ('costs p , was q')
Usage: python apply_stage4.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
Ev, E = os.path.join(root, "trn", "env.py"), os.path.join(root, "trn", "evolve.py")


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


NEW = '''
# ---------------- stage 4: harder reasoning + generalisation (STAGE4_PATCH) ----------------
def uniq_names(rng, split, n, syl=(2, 3), avoid=()):
    out = []
    while len(out) < n:
        nm = make_name(rng, split, syl)
        if nm not in out and nm not in avoid:
            out.append(nm)
    return out


class Hop2(Env):
    """two-hop facts: the asked-about package is named only through another package's dependency."""
    family, stage, max_steps = "hop2", 4, 1

    def __init__(self, rng, split):
        super().__init__(rng, split)
        n = rng.randint(2, 3)
        names = uniq_names(rng, split, 2 * n)
        pk, deps = names[:n], names[n:]
        kind = rng.choice(["version", "costs"])
        vals = [str(v) for v in rng.sample(range(2, 99), n)]
        q = rng.randrange(n)
        lines = [f"{p} needs {d} ." for p, d in zip(pk, deps)] + [f"{d} {kind} {v} ." for d, v in zip(deps, vals)]
        rng.shuffle(lines)
        what = "the version" if kind == "version" else "the price"
        self.first = f"question : what is {what} of the package that {pk[q]} needs ? output : {' '.join(lines)} answer :"
        self.answer = [vals[q]]

    def oracle(self):
        return "answer", self.answer


CITY_KEYS = ["population", "mayor", "river"]


class Chain(Env):
    """look up X's capital, then look up the capital's <key>: the key must be carried across observations."""
    family, stage, max_steps = "chain", 4, 5

    def __init__(self, rng, split):
        super().__init__(rng, split)
        self.c, self.cap, self.lang = uniq_names(rng, split, 3, (2, 2))
        self.city = {"population": str(rng.randint(2, 90)), "mayor": rng.choice(PEOPLE), "river": make_name(rng, split, (2, 2))}
        self.key = rng.choice(CITY_KEYS)
        self.first = f"task : look up the capital of {self.c} , then look up its {self.key} ."
        self.answer = [self.city[self.key]]
        self.order = rng.sample(CITY_KEYS, 3)
        self.phase = 0

    def tool(self, a, args):
        if a == "lookup" and args == [self.c]:
            self.phase = max(self.phase, 1)
            return f"{self.c} : language {self.lang} ; capital {self.cap} ." if self.rng.random() < 0.5 else \\
                   f"{self.c} : capital {self.cap} ; language {self.lang} ."
        if a == "lookup" and args == [self.cap] and self.phase >= 1:
            self.phase = 2
            unit = {"population": " million", "mayor": "", "river": ""}
            return f"{self.cap} : " + " ; ".join(f"{k} {self.city[k]}{unit[k]}" for k in self.order) + " ."
        if a == "lookup":
            return "error : not in table . task : " + self.first
        return super().tool(a, args)

    def oracle(self):
        return [("lookup", [self.c]), ("lookup", [self.cap]), ("answer", self.answer)][self.phase]


TP2_FRAMES = ["please find {x} .", "i need {x} .", "can you get me {x} ?", "help me with {x} .", "{x} , thanks .", "my boss wants {x} ."]
TP2_PHRASES = {"calculator": ["the product of {a} and {b}", "{a} plus {b} worked out", "the sum of {a} and {b}", "{a} minus {b} done fast"],
               "lookup": ["the capital of {n}", "how many people live in {n}", "the language spoken in {n}"],
               "memory": ["a note that keeps the code {a}", "the code {a} stored for later", "a place to keep {n}"],
               "search": ["a package for {f} files", "a tool that converts {f} files", "something to parse {f} files"],
               "browser": ["the site {n} opened", "the downloads page of {n}", "the home page of {n}"]}


class ToolPick2(Env):
    """tool choice under unseen (frame, phrase) combinations: compositional generalisation."""
    family, stage, max_steps = "toolpick2", 4, 1

    def __init__(self, rng, split):
        super().__init__(rng, split)
        while True:
            tool = rng.choice(list(TP2_PHRASES))
            fi, pi = rng.randrange(len(TP2_FRAMES)), rng.randrange(len(TP2_PHRASES[tool]))
            if _is_test(f"tp2|{fi}|{tool}|{pi}|x") == (split == "test"):
                break
        x = TP2_PHRASES[tool][pi].format(a=rng.randint(2, 99), b=rng.randint(2, 99), n=make_name(rng, split), f=rng.choice(FORMATS))
        req = TP2_FRAMES[fi].format(x=x)
        opts = list(TP2_PHRASES)
        rng.shuffle(opts)
        self.first = f"request : {req} options : {' , '.join(opts)} . which tool :"
        self.answer = [tool]

    def oracle(self):
        return "answer", self.answer


class Instr2(Env):
    """corrections with three names, double corrections and unseen correction phrasings."""
    family, stage, max_steps = "instr2", 4, 1

    def __init__(self, rng, split):
        super().__init__(rng, split)
        x, y, z = uniq_names(rng, split, 3)
        v = rng.choice(["install", "open", "delete", "download"])
        form = rng.randrange(4)
        if form == 0:
            text, ans = f"{v} {x} . no wait , {v} {y} . actually , go back to {x} .", x
        elif form == 1:
            text, ans = f"{v} {y} . sorry , i meant {x} .", x
        elif form == 2:
            text, ans = f"do not {v} {x} or {y} . {v} {z} .", z
        else:
            text, ans = f"{v} {x} , not {y} and not {z} .", x
        opts = [x, y, z]
        rng.shuffle(opts)
        self.first = f"instruction : {text} which one should you {v} ? answer {opts[0]} , {opts[1]} , or {opts[2]} :"
        self.answer = [ans]

    def oracle(self):
        return "answer", self.answer


class WebDeep(Env):
    """deeper site navigation (1-3 links below the shop), more sections, decoy old prices."""
    family, stage, max_steps = "webdeep", 4, 9
    L1 = ["toys", "tools", "books", "food", "games"]
    L2 = ["new", "used", "sale", "top"]

    def __init__(self, rng, split):
        super().__init__(rng, split)
        self.site = make_name(rng, split, (2, 2))
        shop = rng.choice(["shop", "store", "market"])
        self.secs = rng.sample(["news", "about", "help", "blog", "contact", "jobs", "faq"], 4) + [shop]
        rng.shuffle(self.secs)
        depth = rng.randint(1, 3)
        self.path = [shop] + ([rng.choice(self.L1)] if depth >= 2 else []) + ([rng.choice(self.L2)] if depth >= 3 else [])
        names = uniq_names(rng, split, 6, (2, 2), avoid=(self.site,))
        self.items = [(n, str(p), str(q)) for n, p, q in zip(names, rng.sample(range(2, 500), 6), rng.sample(range(2, 500), 6))]
        self.item = rng.choice(self.items)
        self.first = f"task : on the site {self.site} find the price of {self.item[0]} ."
        self.loc, self.links = None, []

    def _page(self, loc):
        it = self.item[0]
        if loc == "home":
            return f"site {self.site} : links : " + " ; ".join(self.secs) + " .", list(self.secs)
        if loc in self.path:
            i = self.path.index(loc)
            if i == len(self.path) - 1:
                return (f"page {loc} : " + " ".join(f"{n} costs {p} , was {q} ;" for n, p, q in self.items) + f" find {it} .", [])
            kids = self.L1 if i == 0 else self.L2
            return f"page {loc} : links : " + " ; ".join(kids) + f" . {it} is in {self.path[i + 1]} .", list(kids)
        sib = self.L1 if loc in self.L1 else self.L2 if loc in self.L2 else self.secs
        return f"page {loc} : nothing about {it} here . links : " + " ; ".join(sib) + " .", list(sib)

    def tool(self, a, args):
        if a == "open" and len(args) == 1:
            p = args[0]
            if p == self.site:
                self.loc = "home"
            elif p in self.links:
                self.loc = p
            else:
                return "error : no such page . task : " + self.first
            txt, self.links = self._page(self.loc)
            return txt
        return super().tool(a, args)

    @property
    def answer(self):
        return [self.item[1]]

    def oracle(self):
        if self.loc is None:
            return "open", [self.site]
        if self.loc == self.path[-1]:
            return "answer", self.answer
        if self.loc in self.path:
            return "open", [self.path[self.path.index(self.loc) + 1]]
        for p in self.path:
            if p in self.links:
                return "open", [p]
        return "open", [self.site]
'''

e = open(Ev).read()
if "STAGE4_PATCH" not in e:
    e = sub(e, "\n\nFAMILIES = {c.family: c for c in (Facts, Instr, ToolPick, Calc, Lookup, Memory, Install, Download, Web, WebInstall, Research)}",
            NEW + "\n\nFAMILIES = {c.family: c for c in (Facts, Instr, ToolPick, Calc, Lookup, Memory, Install, Download, Web, WebInstall, Research,\n"
                  "                                     Hop2, Chain, ToolPick2, Instr2, WebDeep)}")
    e = sub(e, "for s in range(4)}", "for s in range(5)}")
    e = sub(e, '"autonomous (simulated) internet + extended tasks"]', '"autonomous (simulated) internet + extended tasks",\n               "multi-hop reasoning, carried goals, compositional generalisation"]')
    open(Ev, "w").write(e)

v = open(E).read()
v = sub(v, 'if st["stage"] < 3 and (competent or stuck):', 'if st["stage"] < max(STAGES) and (competent or stuck):')
open(E, "w").write(v)
print("stage4 patch applied to", root)
