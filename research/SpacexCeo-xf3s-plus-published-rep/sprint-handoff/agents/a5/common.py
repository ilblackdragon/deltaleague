"""Shared helpers for a5 (teacher forcing on CPU, labelling of re-check / progress / close positions)."""
import re, json, torch
RECHECK_VERB = r"(check|double-check|double|re-check|recheck|verify|re-verify|re-read|reread|re-examine|reexamine|re-evaluate|reevaluate|reconsider|look|try|assume|think|go|revisit|see|confirm|test|review|recall|rethink|re-calculate|recalculate|re-think|retry|step|pause|backtrack|examine)"
PROG_VERB = r"(calculate|compute|write|solve|simplify|find|substitute|expand|plug|start|denote|define|evaluate|apply|use|break|set|determine|count|list|identify|formalize|draft|finalize|summarize|construct|structure|put|combine|multiply|add|divide|subtract|factor|convert|express|calculating|proceed|organize|format)"
RE_LET = re.compile(r"^(Let's|Let me|let's|let me)\s+(\S+)")
RE_RECHECK1 = re.compile(r"^(Wait|wait|Hmm|hmm|Alternatively|Actually|Maybe|maybe|Perhaps|What if|Or maybe|Or perhaps|Is there|Is it|Is this|Could|Did I|Hold on|However|But|Or|Hmm,|Wait,|Am I|Are there|Why)\b")
RE_PROG1 = re.compile(r"^(So|Therefore|Thus|Hence|Now|Next|Then|The|This|We|Step|Final|First|Second|Third|Finally|Since|Given|Using|Substitut\w*|Calculat\w*|Compute|Answer|Conclusion|Result|Combining|Plugging|Simplify\w*|Multiply\w*|Adding|Total|Option)\b")
SEG = re.compile(r"\n+[ \t]*(?:(?:[*\-]|\d+\.)[ \t]+)?(?:\*\*?)?")

def load_model(path, threads=6):
    torch.set_num_threads(threads)
    from transformers import AutoConfig, AutoTokenizer, AutoModelForImageTextToText
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float32).eval()
    return tok, m

def lm(m):
    return m.model.language_model

def segments(think):
    """yield (char_offset_of_first_word, label, first_words). label in recheck/progress/None"""
    for mt in SEG.finditer(think):
        st = mt.end(); seg = think[st:st + 60]
        if not seg or seg[0].isspace(): continue
        ml = RE_LET.match(seg)
        if ml:
            verb = ml.group(2).strip(".,:;!?'\"*")
            off = st + ml.start(2)  # decision is made at the verb token
            if re.fullmatch(RECHECK_VERB, verb): yield off, 'recheck', seg[:30]
            elif re.fullmatch(PROG_VERB, verb): yield off, 'progress', seg[:30]
            continue
        if RE_RECHECK1.match(seg): yield st, 'recheck', seg[:30]
        elif RE_PROG1.match(seg): yield st, 'progress', seg[:30]

def lines(think, minlen=12):
    """line starts: 'groove' if the stripped line already appeared verbatim earlier in the trace, else 'novel'"""
    seen = set(); off = 0
    for ln in think.split('\n'):
        st = off + (len(ln) - len(ln.lstrip())); key = re.sub(r'\d+', '#', ln.strip()); off += len(ln) + 1
        if len(key) < minlen: continue
        yield st, ('groove' if key in seen else 'novel'), key[:30]
        seen.add(key)

def build(tok, prompt, text, max_len, with_lines=False):
    """returns ids (1D), gen_start index, offsets of gen tokens relative to gen text, list of (pos,label)"""
    pre = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pid = tok(pre, add_special_tokens=False)['input_ids']
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    gid, offs = enc['input_ids'], enc['offset_mapping']
    ids = (pid + gid)[:max_len]; g0 = len(pid); ng = len(ids) - g0
    # map char offset -> token index (token whose span contains the char, adjusting for leading space)
    starts = [o[0] for o in offs[:ng]]; ends = [o[1] for o in offs[:ng]]
    import bisect
    lab = []
    ci = text.find('</think>')
    think = text if ci < 0 else text[:ci]
    items = list(segments(think)) + (list(lines(think)) if with_lines else [])
    for off, l, w in items:
        k = bisect.bisect_right(starts, off) - 1  # token that contains or starts at off
        if k < 1 or k >= ng: continue
        if ends[k] <= off: continue
        lab.append((g0 + k - 1, l, w))  # position predicting token k
    if ci >= 0:
        k = bisect.bisect_left(starts, ci)
        if k < ng and k >= 1: lab.append((g0 + k - 1, 'close', ''))
    return torch.tensor(ids), g0, lab
