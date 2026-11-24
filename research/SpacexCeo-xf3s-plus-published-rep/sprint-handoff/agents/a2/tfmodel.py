"""Load base Qwen3.5-0.8B on CPU and rebuild the a9340e14 edit in memory (pure scale 0.5 on 7 Wait/Hmm rows,
</think> x1.3, final norm (1+w)*1.4). Exposes load(), hidden(ids)->post-norm final hidden states (fp32)."""
import torch, time
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
B='/home/user/delta/exp-base/model'
NEW7=[' wait','wait',' Wait','Wait','Hmm',' hmm',' Hmm']
def load(edit='new'):
    torch.set_num_threads(4)
    tok=AutoTokenizer.from_pretrained(B)
    m=Qwen3_5ForConditionalGeneration.from_pretrained(B, dtype=torch.float32); m.eval()
    lm=m.model.language_model
    E=lm.embed_tokens.weight.data
    V=tok.get_vocab()
    if edit=='new':
        for s in NEW7: E[V[s.replace(' ','Ġ')]]*=0.5
        E[V['</think>'] if '</think>' in V else tok.convert_tokens_to_ids('</think>')]*=1.3
        lm.norm.weight.data=(1.4*(1+lm.norm.weight.data.float())-1)
    return tok,m
@torch.no_grad()
def hidden(m, ids):
    out=m.model.language_model(input_ids=torch.tensor([ids]), use_cache=False)
    return out.last_hidden_state[0]
