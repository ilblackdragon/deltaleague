import re
def letter(ans):
    """Format-tolerant MC extraction: boxed, then 'answer is X', then **X, then a bare letter line."""
    b=[]; i=0
    while True:
        j=ans.find("\\boxed{",i)
        if j<0: break
        k=ans.find("}",j); b.append(ans[j+7:k]); i=j+7
    if b:
        m=re.findall(r"[A-J]",b[-1].replace("\\text",""))
        if m: return m[0]
    m=re.findall(r"(?:answer is|Answer:?)\s*[:\*\(\s]*([A-J])\b",ans)
    if m: return m[-1]
    m=re.findall(r"\*\*\(?([A-J])\)?[\.\*:]",ans)
    if m: return m[-1]
    m=re.findall(r"^\s*\(?([A-J])\)?\s*$",ans.strip(),re.M)
    return m[-1] if m else ""
