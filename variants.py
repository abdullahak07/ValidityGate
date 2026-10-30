r"""
variants.py - option-order variants of multiple-choice questions (TNNLS revision, steps 2 and held-out eval).

A question stores its options twice: in s["options"] (letter -> text) and as lines "A. text" inside s["question"].
rotate(s, k)   : options rotated by k positions (letters stay A, B, C...; contents move; gold letter recomputed)
reverse(s)     : options in reverse order (for >= 3 options this order is never a rotation, so it stays held out
                 when training uses rotations only)
augment4(lst)  : 4 training copies per question = rotations 0..n-1, repeated to fill 4 (n = number of options)
All functions return None / skip a question whose text does not contain its option lines in the expected form.
"""
LETTERS = ["A", "B", "C", "D", "E"]


def _parts(s):
    opts = s.get("options") or {}
    letters = [L for L in LETTERS if L in opts]
    if len(letters) < 2:
        return None
    lines = str(s["question"]).split("\n")
    try:
        pos = [lines.index(f"{L}. {opts[L]}") for L in letters]
    except ValueError:
        return None
    if pos != list(range(pos[0], pos[0] + len(letters))):
        return None
    return opts, letters, lines, pos[0]


def _with_order(s, new_texts):
    p = _parts(s)
    if p is None:
        return None
    opts, letters, lines, start = p
    gold_text = opts.get(str(s["answer"]).strip().upper())
    if gold_text is None:
        return None
    new_opts = dict(zip(letters, new_texts))
    gold = [L for L in letters if new_opts[L] == gold_text]
    if len(gold) != 1:                       # duplicate option texts: ambiguous
        return None
    new_lines = lines[:start] + [f"{L}. {new_opts[L]}" for L in letters] + lines[start + len(letters):]
    out = dict(s)
    out.update(question="\n".join(new_lines), options=new_opts, answer=gold[0])
    return out


def rotate(s, k):
    p = _parts(s)
    if p is None:
        return None
    opts, letters, _, _ = p
    n = len(letters)
    if k % n == 0:
        return None
    texts = [opts[L] for L in letters]
    return _with_order(s, texts[k % n:] + texts[:k % n])


def reverse(s):
    p = _parts(s)
    if p is None or len(p[1]) < 3:           # 2 options: reversal == rotation, not held out
        return None
    opts, letters, _, _ = p
    texts = [opts[L] for L in letters]
    rev = texts[::-1]
    if any(rev == texts[k:] + texts[:k] for k in range(len(texts))):
        return None                          # e.g. options (x, y, x): reversed order equals a trained rotation
    return _with_order(s, rev)


def augment4(samples):
    out = []
    for s in samples:
        p = _parts(s)
        if p is None:
            out += [s] * 4
            continue
        n = len(p[1])
        for c in range(4):
            k = c % n
            out.append(s if k == 0 else (rotate(s, k) or s))
    return out
