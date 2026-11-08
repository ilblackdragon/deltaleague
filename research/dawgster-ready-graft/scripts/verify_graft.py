import json,torch,sys
from transformers import AutoTokenizer, AutoModelForCausalLM
cand=sys.argv[1]; tok=AutoTokenizer.from_pretrained("models/leader")
items={it["id"]:it for it in json.load(open("runs/proxy.json"))}
rows=[json.loads(l) for l in open("runs/leader/seed2.jsonl")]; rows=[r for r in rows if r["text"].count("</think>")==1 and r["score"]][-8:]
res={}
for name in ["leader",cand]:
    m=AutoModelForCausalLM.from_pretrained(f"models/{name}",dtype=torch.bfloat16).cuda().eval(); out=[]
    for r in rows:
        p=tok.apply_chat_template([{"role":"user","content":items[r["id"]]["prompt"]}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
        pi=tok(p,add_special_tokens=False).input_ids; ti=tok(r["text"].split("</think>")[0],add_special_tokens=False).input_ids[:4000]
        with torch.no_grad(): lg=m(torch.tensor([pi+ti]).cuda()).logits[0,len(pi):].float()
        out.append((lg[:,248069].cpu(), lg.log_softmax(-1)[:,248069].cpu(), lg.argmax(-1).cpu()))
    res[name]=out; del m; torch.cuda.empty_cache()
for (a,la,aa),(b,lb,ab) in zip(res["leader"],res[cand]):
    n=len(a); d=b-a; mid=slice(int(.2*n),int(.7*n))
    print(f"n={n} dlogit mid {d[mid].mean():+.2f} last150 {d[-150:].mean():+.2f} final {d[-1]:+.2f} | P(close@final) {la[-1].exp():.3f}->{lb[-1].exp():.3f} | argmax changed {(aa!=ab).float().mean()*100:.2f}%")
