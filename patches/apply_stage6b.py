"""Patch: todo 'save the code N as K' accepts both argument orders, like research (STAGE6B_PATCH). Idempotent.
Bug (2026-11-27): the agent emits pointer arguments in text order ([N, K]) and the carry note keeps text order too;
Todo required [K, N] exactly, so every todo episode failed at the save step (stage-6 baseline todo 0.0).
Usage: python apply_stage6b.py <dir containing trn/>"""
import sys, os
root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
Ev = os.path.join(root, "trn", "env.py")
e = open(Ev).read()
old = '''            if args == [self.key, self.code]:
                self.ok.add("save")'''
new = '''            if sorted(args) == sorted([self.key, self.code]):      # STAGE6B_PATCH: either order
                self.ok.add("save")'''
if "STAGE6B_PATCH" not in e:
    assert e.count(old) == 1
    e = e.replace(old, new)
    e = e.replace('''            return "remember", [self.key, self.code]
        return ("answer", self.answer) if self.opened''', '''            return "remember", [self.code, self.key]           # STAGE6B_PATCH: text order
        return ("answer", self.answer) if self.opened''')
    assert "[self.code, self.key]" in e
    open(Ev, "w").write(e)
print("stage6b patch applied to", root)
