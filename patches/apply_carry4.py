"""Patch on top of apply_stage5: carry >= 3 -- the note is written in TEXT order (oracle labels and inference),
so its layout is consistent ('note : file site field .'). Idempotent.

Trace (stage 5, carry=2, 2026-11-26): in multi the note held the right three words but in pointer-score order
('size ruqui quiqui', 'talri size hilxa', 'owner dungar seneki'), and 'open' copied the field word with one
name ('open size ruqui'). A fixed layout lets the unary pointer learn which note words are arguments.
Usage: python apply_carry4.py <dir containing trn/>"""
import sys, os

root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/TRN"
A = os.path.join(root, "trn", "agent.py")


def sub(src, old, new):
    if new in src:
        return src
    assert src.count(old) == 1, old
    return src.replace(old, new)


a = open(A).read()
if "CARRY4_PATCH" not in a:
    a = sub(a, '''                         if u not in s.note][:2 + self.carry2]''', '''                         if u not in s.note][:2 + self.carry2]
            if self.carry3:                              # CARRY4_PATCH: text order
                s.ctarget.sort(key=lambda u: s.obs_words.index(u) if u in s.obs_words else 10 ** 6)''')
    a = sub(a, '''        self.carry2 = int((attn or {}).get("carry") or 0) >= 2           # STAGE5_PATCH: note may carry arguments''',
            '''        self.carry2 = int((attn or {}).get("carry") or 0) >= 2           # STAGE5_PATCH: note may carry arguments
        self.carry3 = int((attn or {}).get("carry") or 0) >= 3           # CARRY4_PATCH: note written in text order''')
    a = sub(a, '''                                    s.carry.append(s.obs_words[w_])
                    obs = s.env.act(a, args)''', '''                                    s.carry.append(s.obs_words[w_])
                            if self.carry3:                  # CARRY4_PATCH
                                s.carry.sort(key=lambda u: s.obs_words.index(u))
                    obs = s.env.act(a, args)''')
    open(A, "w").write(a)

Mu = os.path.join(root, "trn", "mutate.py")
m = open(Mu).read()
if "CARRY4_PATCH" not in m:
    m = sub(m, '''{0: 1, 1: 2, 2: 1})[cur]   # STAGE5_PATCH: carry 2 = args too''',
            '''{0: 1, 1: 2, 2: 3, 3: 2})[cur]   # STAGE5_PATCH: carry 2 = args too; CARRY4_PATCH: 3 = text-ordered''')
    open(Mu, "w").write(m)
print("carry4 patch applied to", root)
