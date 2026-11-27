"""Patch on top of apply_occrec (OCCREC2_PATCH): the option-list (last) occurrence of a repeated word gets no
recency bonus, nor does a word that occurs once; recency ranks the in-text mentions of repeated words only. A/B (2026-11-27): with the bonus on the option list,
aff labels collapsed instr 0.98 -> 0.84 (every option word's last occurrence sits at the very end). Idempotent."""
import sys, os
root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
A = os.path.join(root, "trn", "agent.py")
a = open(A).read()
old = '''        z[ok] = (u[ok] - u[ok].mean()) / (u[ok].std() + 1e-6) + beta * np.arange(len(u))[ok] / max(1, len(u))
        return self._occ_pool(z, ws)'''
new = '''        z[ok] = (u[ok] - u[ok].mean()) / (u[ok].std() + 1e-6) + beta * np.arange(len(u))[ok] / max(1, len(u))
        last, cnt = {}, {}                           # OCCREC2_PATCH: no recency bonus for the option-list occurrence
        for i, w in enumerate(ws[:len(u)]):
            cnt[w] = cnt.get(w, 0) + 1; last[w] = i
        for i, w in enumerate(ws[:len(u)]):      # bonus only for NON-final occurrences of repeated words
            if ok[i] and (cnt[w] == 1 or i == last[w]):
                z[i] -= beta * i / max(1, len(u))
        return self._occ_pool(z, ws)'''
if "OCCREC2_PATCH" not in a:
    assert a.count(old) == 1
    open(A, "w").write(a.replace(old, new))
print("occrec2 patch applied to", root)
