"""Patch on top of apply_carry2: rare-word counting ignores the note prefix (a noted word counts as linked when it
occurs once in the text), so a carried note can create a bridge without destroying an existing one. Idempotent.
A/B before this fix: chain 66 -> 83% but webdeep 99 -> 61% (noted item name occurred 3x -> no longer 'rare')."""
import sys, os
root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
A = os.path.join(root, "trn", "agent.py")


def sub(src, old, new, count=1):
    if new in src:
        return src
    assert src.count(old) == count, (old, src.count(old))
    return src.replace(old, new)


a = open(A).read()
if "CARRY3_PATCH" not in a:
    a = sub(a, '''def chain_info(ws, args, maxh):''', '''def rare_fn(ws):
    """CARRY3_PATCH: word -> is a link word (twice in the text, or once in the text + in the note)."""
    k = ws.index(".") + 1 if len(ws) > 2 and ws[0] == "note" and ws[1] == ":" and "." in ws else 0
    noted = set(ws[2:k - 1]) if k else set()
    cnt = collections.Counter(ws[k:])
    return lambda v: v not in PUNCT and (cnt[v] == 2 or (cnt[v] == 1 and v in noted))


def chain_info(ws, args, maxh):''')
    a = sub(a, '''    ce, cs = clause_ends(ws), clause_starts(ws)
    cnt = collections.Counter(ws)
    argset = set(args)
    depth = {cs[p]: 0 for p in pos}''', '''    ce, cs = clause_ends(ws), clause_starts(ws)
    rare = rare_fn(ws)
    argset = set(args)
    depth = {cs[p]: 0 for p in pos}''')
    a = sub(a, '''                if v in PUNCT or v in argset or cnt[v] != 2:
                    continue''', '''                if v in argset or not rare(v):
                    continue''')
    a = sub(a, '''        cnt = collections.Counter(ws)
        wv = np.array([cw[cid[i]] * (1.0 - m[i]) if (cnt[ws[i]] == 2 and ws[i] not in PUNCT) else 0.0 for i in range(n)])''',
            '''        rare = rare_fn(ws)
        wv = np.array([cw[cid[i]] * (1.0 - m[i]) if rare(ws[i]) else 0.0 for i in range(n)])''')
    open(A, "w").write(a)
print("carry3 patch applied to", root)
