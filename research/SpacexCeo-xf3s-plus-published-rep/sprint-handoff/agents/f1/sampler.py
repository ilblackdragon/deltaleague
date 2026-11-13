"""Exact vLLM-style sampler on stored teacher-forced positions, under an arbitrary edit theta of the family.
theta = dict(g=1.0, s={token_name: s}, mode='scale'|'mean', c=1.0 (</think> row scale), bias={token_name: b}, extra_scale={name: s})
Returns per-position probabilities of: close, hedge7, hedge13 (both under the edit), actual token, entropy, nucleus size."""
import numpy as np, json, os
H7 = [' wait', 'wait', ' Wait', 'Wait', 'Hmm', ' hmm', ' Hmm']
H11 = [' wait', ' Maybe', ' Wait', 'Wait', 'Maybe', ' Actually', ' Alternatively', 'Actually', 'Hmm', ' Hmm', 'Alternatively']
H13 = H7 + [' Maybe', 'Maybe', ' Actually', 'Actually', ' Alternatively', 'Alternatively']
SUBS = [' But', 'But', ' Let', ' However', ' Hold', ' Double', ' Oh', ' Alright', 'Alright', ' Perhaps']
PEN = 1.5; TOPK = 20; TOPP = 0.95
class Trace:
    def __init__(self, fn, meta):
        d = np.load(fn); self.d = d
        self.watch = meta['watch']; self.wid = np.array(meta['wid']); self.mu_sq = meta['mu_sq']
        self.widx = {n: i for i, n in enumerate(self.watch)}
        self.topv = d['topv'].astype(np.float64); self.topi = d['topi']; self.topseen = d['topseen']
        self.wz = d['wz'].astype(np.float64); self.wseen = d['wseen']; self.mlog = d['mlog'].astype(np.float64)
        self.gid = d['gid']; self.closed = bool(d['closed']); self.n = self.topv.shape[0]
        # union candidate set per position: top64 + watch rows (watch rows not already in top64)
        self.in_top = np.zeros((self.n, len(self.wid)), bool)
        for k, w in enumerate(self.wid): self.in_top[:, k] = (self.topi == w).any(1)
    def logits(self, th):
        """edited, penalised logits for candidate set: returns (L_top (n,64), L_w (n,W)) and flags"""
        g = th.get('g', 1.0); mode = th.get('mode', 'scale'); c = th.get('c', 1.0)
        s = dict(th.get('s', {})); bias = th.get('bias', {})
        zt = self.topv.copy(); zw = self.wz.copy()
        # per-row scales applied to watch rows (and to the same ids inside top64)
        scale = {}
        for name, sv in s.items(): scale[name] = sv
        if c != 1.0: scale['</think>'] = scale.get('</think>', 1.0) * c
        for name, sv in scale.items():
            k = self.widx[name]; z = self.wz[:, k]
            if mode == 'mean' and name in s:   # mean interpolation only for the dampened set, close row is pure scale
                znew = s[name] * z + (1 - s[name]) * self.mlog
                if name == '</think>' and c != 1.0: znew = znew * c
            else: znew = sv * z
            zw[:, k] = znew
            msk = self.topi == self.wid[k]; zt[msk] = np.broadcast_to(znew[:, None], zt.shape)[msk]
        for name, b in bias.items():
            k = self.widx[name]; add = b * self.mlog / self.mu_sq
            zw[:, k] += add; msk = self.topi == self.wid[k]; zt[msk] = (zt + add[:, None])[msk]
        Lt = g * zt - PEN * self.topseen; Lw = g * zw - PEN * self.wseen
        return Lt, Lw
    def probs(self, th):
        Lt, Lw = self.logits(th); n = self.n
        # candidate set: top64 plus watch rows not in top64 (set their top-copy to -inf to avoid double counting)
        Lw_extra = np.where(self.in_top, -np.inf, Lw)
        L = np.concatenate([Lt, Lw_extra], 1)                     # (n, 64+W)
        ids = np.concatenate([self.topi, np.broadcast_to(self.wid, (n, len(self.wid)))], 1)
        # top-k
        order = np.argsort(-L, 1); kth = np.take_along_axis(L, order[:, TOPK - 1:TOPK], 1)
        L = np.where(L >= kth, L, -np.inf)
        p = np.exp(L - L.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        # top-p (vLLM: sort desc, keep tokens while cumsum - p < top_p)
        o = np.argsort(-p, 1); ps = np.take_along_axis(p, o, 1); cs = np.cumsum(ps, 1)
        keep_sorted = (cs - ps) < TOPP
        keep = np.zeros_like(keep_sorted); np.put_along_axis(keep, o, keep_sorted, 1)
        p = np.where(keep, p, 0.0); p /= p.sum(1, keepdims=True)
        out = {}
        pw = np.zeros((n, len(self.wid)))
        for k, w in enumerate(self.wid): pw[:, k] = (p * (ids == w)).sum(1)
        out['pw'] = pw
        out['p_close'] = pw[:, self.widx['</think>']]
        out['p_h7'] = sum(pw[:, self.widx[x]] for x in H7)
        out['p_h13'] = sum(pw[:, self.widx[x]] for x in H13)
        out['p_subs'] = sum(pw[:, self.widx[x]] for x in SUBS if x in self.widx)
        act = (p * (ids == self.gid[:, None])).sum(1); out['p_act'] = act
        with np.errstate(divide='ignore', invalid='ignore'):
            out['H'] = -(np.where(p > 0, p * np.log(p), 0)).sum(1)
        out['nuc'] = (p > 0).sum(1)
        return out
def load_dir(d):
    meta = json.load(open(os.path.join(d, 'meta.json')))
    return {f[:-4]: Trace(os.path.join(d, f), meta) for f in sorted(os.listdir(d)) if f.endswith('.npz')}
NEWLEADER = dict(g=1.4, s={k: 0.5 for k in H7}, mode='scale', c=1.3)
LEADER1 = dict(g=1.4, s={k: 0.6 for k in H11}, mode='mean')
BASE = dict()
