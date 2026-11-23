"""Batched CPU sampler for Qwen3.5 checkpoints, mirroring the Delta League evaluation sampler.

Sampler (same order of operations as vLLM): presence penalty on tokens already generated
in this completion -> temperature -> top-k -> top-p (on the top-k-renormalised
distribution) -> sample. Each sequence draws from its own pre-generated uniform stream,
so its randomness does not depend on which other sequences share its batch, and two
checkpoints can be compared with common random numbers.

Throughput on a CPU-only VM comes from three things: one thread per process (threads
are pathological in this VM), several samples per forward pass, and a float32 checkpoint
that every worker process memory-maps (see to_fp32.py).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import hashlib
import warnings

warnings.filterwarnings("ignore")
import torch
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen_module
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
torch.set_num_threads(1)

THINK_OPEN, THINK_CLOSE = 248068, 248069
EOS_IDS = (248046, 248044)  # <|im_end|>, <|endoftext|>
PAD_ID = 248044
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
# Bytes of full-attention key/value cache per token: 6 layers * (K + V) * 2 heads * 256 dims * 4 bytes.
KV_BYTES_PER_TOKEN = 6 * 2 * 2 * 256 * 4


class BufferLayer(DynamicLayer):
    """Key/value cache layer backed by preallocated buffers that grow geometrically.

    The stock layer concatenates on every token, which reallocates the whole cache each
    step. This one writes new states in place and hands out views, so a decode step
    allocates nothing for the cache. Values are identical; only the storage differs.
    """

    GROW = 1.25
    reserve = 0  # capacity to allocate up front when the final prefill length is known

    def lazy_initialization(self, key_states, value_states):
        self.dtype, self.device = key_states.dtype, key_states.device
        self._k = self._v = None
        self._len = 0
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        need = self._len + key_states.shape[-2]
        if self._k is None or need > self._k.shape[-2]:
            old = 0 if self._k is None else self._k.shape[-2]
            b, h, _, d = key_states.shape
            k = torch.empty((b, h, max(need, self.reserve, int(old * self.GROW) + 64), d), dtype=self.dtype)
            v = torch.empty_like(k)
            if self._len:
                k[:, :, :self._len] = self._k[:, :, :self._len]
                v[:, :, :self._len] = self._v[:, :, :self._len]
            self._k, self._v = k, v
        self._k[:, :, self._len:need] = key_states
        self._v[:, :, self._len:need] = value_states
        self._len = need
        return self._k[:, :, :need], self._v[:, :, :need]

    def get_seq_length(self):
        return self._len if self.is_initialized else 0

    def reorder_cache(self, beam_idx):
        if self.get_seq_length() > 0:
            self._k = self._k.index_select(0, beam_idx)
            self._v = self._v.index_select(0, beam_idx)

    batch_select_indices = reorder_cache

    def reset(self):
        self._k = self._v = None
        self._len = 0
        self.is_initialized = False

    def crop(self, tokens_to_remove):
        raise NotImplementedError("BufferLayer does not support cropping")

    def batch_repeat_interleave(self, repeats):
        raise NotImplementedError("BufferLayer does not support repeat_interleave")


def grouped_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Grouped-query attention without materialising repeated keys and values.

    Mathematically the same as transformers' eager_attention_forward for this model
    (8 query heads sharing 2 key/value heads): query head h uses key/value head h // 4.
    The stock function copies the whole key and value cache 4 times on every call.
    """
    b, heads, q_len, dim = query.shape
    groups = key.shape[1]
    q = query.reshape(b, groups, (heads // groups) * q_len, dim)
    scores = torch.matmul(q, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = (scores.view(b, groups, heads // groups, q_len, -1) + attention_mask[:, :, None]).view(b, groups, -1, key.shape[2])
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, value).view(b, heads, q_len, dim)
    return out.transpose(1, 2).contiguous(), None


class Sampler:
    def __init__(self, temperature=1.0, top_k=20, top_p=0.95, presence_penalty=1.5, max_tokens=8192):
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.presence_penalty, self.max_tokens = presence_penalty, max_tokens

    def candidates(self, logits, seen):
        """Kept candidates per row: token ids sorted ascending, and their sampling probabilities."""
        z = logits - self.presence_penalty * seen
        if self.temperature != 1.0:
            z = z / self.temperature
        vals, idx = torch.topk(z, self.top_k, dim=-1)  # descending
        p = torch.softmax(vals, dim=-1)
        # vLLM drops a token when the mass of it and everything below it is <= 1 - top_p.
        tail = torch.flip(torch.cumsum(torch.flip(p, [-1]), -1), [-1])
        keep = tail > (1.0 - self.top_p)
        keep[:, 0] = True
        q = p * keep
        q = q / q.sum(-1, keepdim=True)
        ids, order = torch.sort(idx, dim=-1)  # token-id order: a model-independent ordering
        return ids, torch.gather(q, 1, order), torch.gather(vals, 1, order)

    def sample(self, logits, seen, u):
        """Inverse-CDF sampling over the kept candidates with one uniform per row."""
        ids, q, vals = self.candidates(logits, seen)
        cdf = torch.cumsum(q, -1)
        choice = (cdf < u[:, None]).sum(-1)
        last = (q > 0).long().cumsum(-1).argmax(-1)  # last candidate with positive mass
        choice = torch.minimum(choice, last)
        return torch.gather(ids, 1, choice[:, None]).squeeze(1), (ids, q, vals, choice)


class Model:
    def __init__(self, model_dir):
        model_dir = os.path.expanduser(model_dir)
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        full = AutoModelForImageTextToText.from_pretrained(
            model_dir, config=config, local_files_only=True, use_safetensors=True, dtype=torch.float32
        ).eval()
        self.full = full
        self.text = full.model.language_model  # embeddings, 24 layers and the final norm
        self.head = full.lm_head
        self.vocab = config.text_config.vocab_size
        self.fast = False

    def install_fast_paths(self):
        """Switch to the buffered cache and the copy-free grouped attention (verified by test_engine.py)."""
        qwen_module.eager_attention_forward = grouped_attention_forward
        for module in self.text.modules():
            if isinstance(module, qwen_module.Qwen3_5Attention):
                module.config._attn_implementation = "eager"
        self.fast = True
        return self

    def new_cache(self, reserve=0):
        cache = DynamicCache(config=self.text.config)
        if self.fast:
            cache.layers = [BufferLayer() if type(layer) is DynamicLayer else layer for layer in cache.layers]
            for layer in cache.layers:
                if isinstance(layer, BufferLayer):
                    layer.reserve = reserve
        return cache

    def render(self, user_text, thinking=True):
        """Chat-template text for a prompt: a user message, or {"messages": [...], "tools": [...]}.

        The object form is rendered the way the platform's BFCL handler does it (tool schemas
        passed to the chat template), so tool-calling prompts reach the same model states.
        """
        if isinstance(user_text, dict):
            tools = [{"type": "function", "function": f} for f in user_text.get("tools", [])] or None
            return self.tok.apply_chat_template(
                user_text["messages"], tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
            )
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )

    def encode_prompt(self, user_text, thinking=True):
        return self.tok(self.render(user_text, thinking), add_special_tokens=False).input_ids

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    @torch.inference_mode()
    def prefill(self, rows, chunk=256):
        """Left-padded, chunked forward over token-id lists. Returns (cache, mask, last hidden state)."""
        B, T = len(rows), max(len(r) for r in rows)
        ids = torch.full((B, T), PAD_ID, dtype=torch.long)
        mask = torch.zeros((B, T), dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, T - len(r):] = torch.tensor(r, dtype=torch.long)
            mask[i, T - len(r):] = 1
        # Attention scores for a chunk take B * 8 heads * chunk * T floats; keep that near 64 MB.
        chunk = max(16, min(chunk, 2_000_000 // (B * T)))
        past, out = self.new_cache(reserve=T + 192), None
        for a in range(0, T, chunk):
            b = min(T, a + chunk)
            out = self.text(input_ids=ids[:, a:b], attention_mask=mask[:, :b], past_key_values=past, use_cache=True)
            past = out.past_key_values
        return past, mask, out.last_hidden_state[:, -1]

    @torch.inference_mode()
    def step(self, tokens, past, mask):
        """One cached decode step for a batch. Returns (cache, mask, last hidden state)."""
        mask = torch.cat([mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype)], dim=1)
        out = self.text(input_ids=tokens[:, None], attention_mask=mask, past_key_values=past, use_cache=True)
        return out.past_key_values, mask, out.last_hidden_state[:, -1]


def uniform_stream(seed, key, n):
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    g = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") >> 1)
    return torch.rand(n, generator=g, dtype=torch.float64)


class Seq:
    """One completion in progress. `out` holds generated token ids (EOS excluded)."""
    __slots__ = ("key", "meta", "prompt", "out", "u", "finish", "forced")

    def __init__(self, key, prompt, seed, max_tokens, meta=None, out=None):
        self.key, self.prompt, self.meta = key, list(prompt), meta or {}
        self.out = list(out or [])
        self.u = uniform_stream(seed, key, max_tokens + 1)
        self.finish = None
        self.forced = {}  # step index -> token id, for decode-time interventions

    def total(self):
        return len(self.prompt) + len(self.out)


def batch_limit(length_now, max_batch, kv_budget_bytes, headroom=1024):
    """Largest batch whose key/value cache stays within the budget after `headroom` more tokens."""
    return max(2, min(max_batch, kv_budget_bytes // ((length_now + headroom) * KV_BYTES_PER_TOKEN)))


@torch.inference_mode()
def run_batch(model, seqs, sampler, may_suspend, on_finish, kv_budget_bytes, hook=None, bias=None):
    """Decode a batch until every sequence finishes, or suspend it when re-batching is better.

    Returns the unfinished sequences (empty unless suspended). `hook(seq, step, token)` may
    return a replacement token id, which is how decode-time interventions are expressed.
    `bias(seq, step)` may return [(token id, logit delta), ...] added before sampling, which
    emulates an edit that pushes a token without forcing it.
    """
    past, mask, h = model.prefill([s.prompt + s.out for s in seqs])
    seen = torch.zeros((len(seqs), model.vocab), dtype=torch.bool)
    for i, s in enumerate(seqs):
        if s.out:
            seen[i, torch.tensor(sorted(set(s.out)), dtype=torch.long)] = True
    live = list(seqs)
    start_count = len(seqs)
    while live:
        u = torch.tensor([float(s.u[len(s.out)]) for s in live], dtype=torch.float32)
        logits = model.head(h)
        if bias is not None:
            for pos, s in enumerate(live):
                for token_id, delta in bias(s, len(s.out)) or ():
                    logits[pos, token_id] += delta
        tokens, _ = sampler.sample(logits, seen, u)
        keep = []
        for pos, s in enumerate(live):
            t = int(tokens[pos])
            step = len(s.out)
            if step in s.forced:
                t = s.forced[step]
            if hook is not None:
                r = hook(s, step, t)
                if r is not None:
                    t = int(r)
            tokens[pos] = t
            if t in EOS_IDS:
                s.finish = "stop"
            else:
                s.out.append(t)
                if len(s.out) >= sampler.max_tokens:
                    s.finish = "length"
            if s.finish is None:
                keep.append(pos)
            else:
                on_finish(s)
        if not keep:
            return []
        if len(keep) < len(live):
            sel = torch.tensor(keep, dtype=torch.long)
            past.reorder_cache(sel)
            mask, seen, tokens = mask[sel], seen[sel], tokens[sel]
            live = [live[p] for p in keep]
        over_budget = len(live) * mask.shape[1] * KV_BYTES_PER_TOKEN > 1.15 * kv_budget_bytes and len(live) > 2
        if over_budget or (len(live) <= start_count // 2 and may_suspend(len(live))):
            # The sampled tokens are already recorded in `out`; a later prefill replays them.
            return live
        seen[torch.arange(len(live)), tokens] = True
        past, mask, h = model.step(tokens, past, mask)
    return []
