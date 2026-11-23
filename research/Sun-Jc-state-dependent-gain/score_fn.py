"""Score-function statistics for the final-norm gain, from stored completions.

Scaling the final-norm gain by g multiplies every raw logit by g. For a sampled token x
drawn from the evaluation sampler's kept set K (after the presence penalty, top-k and
top-p), the derivative of its log-probability with respect to log g at g = 1 is

    s_t = z_x - sum_{v in K} q_v z_v        (z = raw logits, q = sampling probabilities)

which has mean zero under the sampler at every step, whatever the prompt. So for a reward
R of the whole completion, mean(R * sum_t s_t) over sampled completions is an unbiased
estimate of dE[R] / d log g (REINFORCE), with no need for per-prompt baselines. Sums are
also kept per phase and per step bucket, to see WHERE extra greediness is associated with
success.

collect: python score_fn.py collect --model FP32_DIR --jobs JOBS --run RUN --out OUT [--shard K/P]
report:  python score_fn.py report OUT SCORED_JSONL
"""
import argparse
import glob
import json
import os
import sys

import torch

BUCKETS = [(0, 250), (250, 1000), (1000, 2000), (2000, 4000), (4000, 8193)]


def collect(args):
    from engine import THINK_CLOSE, Model, Sampler
    from scorelib import load_rows

    k, p = (int(x) for x in args.shard.split("/"))
    os.makedirs(args.out, exist_ok=True)
    model = Model(args.model).install_fast_paths()
    sampler = Sampler()
    jobs = {j["id"]: j for j in map(json.loads, open(args.jobs))}
    rows = sorted(load_rows(args.run), key=lambda r: (r["id"], r["sample"]))
    rows = [r for i, r in enumerate(rows) if i % p == k]
    out = []
    with torch.inference_mode():
        for ri, r in enumerate(rows):
            prompt = model.encode_prompt(jobs[r["id"]]["prompt"])
            toks = r["tokens"]
            n, n_prompt = len(toks), len(prompt)
            ids = torch.tensor([prompt + toks], dtype=torch.long)
            first_seen = torch.full((model.vocab,), 1 << 30, dtype=torch.long)
            for i in range(n - 1, -1, -1):
                first_seen[toks[i]] = i
            first_close = toks.index(THINK_CLOSE) if THINK_CLOSE in toks else n
            s_all, h_all = torch.zeros(n), torch.zeros(n)
            past = model.new_cache(reserve=ids.shape[1] + 8)
            last_pos = n_prompt - 1 + n - 1  # state that produced the last generated token
            for a in range(0, last_pos + 1, args.chunk):
                b = min(last_pos + 1, a + args.chunk)
                hid = model.text(input_ids=ids[:, a:b], past_key_values=past, use_cache=True).last_hidden_state[0]
                lo = max(a, n_prompt - 1)
                if lo >= b:
                    continue
                steps = torch.arange(lo - (n_prompt - 1), b - (n_prompt - 1))
                logits = model.head(hid[lo - a:])
                seen = (first_seen[None, :] < steps[:, None]).float()
                cand, q, _ = sampler.candidates(logits, seen)
                z = torch.gather(logits, 1, cand)  # raw logits of the kept candidates
                tgt = torch.tensor([toks[s] for s in steps.tolist()])
                z_x = torch.gather(logits, 1, tgt[:, None]).squeeze(1)
                s_all[steps] = z_x - (q * z).sum(-1)
                h_all[steps] = -(q * torch.log(q.clamp_min(1e-12))).sum(-1)
            rec = {"id": r["id"], "sample": r["sample"], "n": n, "first_close": first_close}
            idx = torch.arange(n)
            for name, phase in (("think", idx < first_close), ("answer", idx > first_close)):
                for lo_b, hi_b in BUCKETS:
                    sel = phase & (idx >= lo_b) & (idx < hi_b)
                    rec[f"s_{name}_{lo_b}"] = float(s_all[sel].sum())
                    rec[f"n_{name}_{lo_b}"] = int(sel.sum())
            rec["s_total"] = float(s_all.sum())
            rec["entropy_total"] = float(h_all.sum())
            rec["var_proxy"] = float(s_all.pow(2).sum())
            out.append(rec)
            if (ri + 1) % 10 == 0 or ri + 1 == len(rows):
                print(f"[score_fn {k}/{p}] {ri + 1}/{len(rows)} rows", file=sys.stderr, flush=True)
                with open(os.path.join(args.out, f"sf-{k}.jsonl"), "w") as f:
                    for rec_ in out:
                        f.write(json.dumps(rec_) + "\n")


def report(out_dir, scored_path):
    recs = [json.loads(l) for f in sorted(glob.glob(os.path.join(os.path.expanduser(out_dir), "sf-*.jsonl"))) for l in open(f)]
    scored = {(s["id"], s["sample"]): s for s in map(json.loads, open(os.path.expanduser(scored_path)))}
    recs = [r for r in recs if (r["id"], r["sample"]) in scored]
    n = len(recs)
    R = torch.tensor([scored[(r["id"], r["sample"])]["score"] for r in recs], dtype=torch.float64)
    capped = torch.tensor([scored[(r["id"], r["sample"])]["outcome"] in ("runaway", "overflow") for r in recs], dtype=torch.float64)
    fam = [scored[(r["id"], r["sample"])]["family"] for r in recs]
    print(f"{n} completions; mean reward {R.mean():.3f}; capped share {capped.mean():.3f}")

    def slope(values, target, label):
        s = torch.tensor(values, dtype=torch.float64)
        prod = (target - target.mean()) * s  # centring the reward lowers variance and keeps the estimate unbiased
        est, se = prod.mean(), prod.std() / n ** 0.5
        print(f"  {label:34s} d/dlog(g) = {100 * est:+8.1f} +- {100 * se:6.1f} points per unit log-gain"
              f"  => per x1.05 step: {100 * est * 0.0488:+5.2f} +- {100 * se * 0.0488:4.2f}   (sum sd {s.std():.1f}, mean {s.mean():+.2f})")

    print("effect of the final-norm gain on the SCORE:")
    slope([r["s_total"] for r in recs], R, "all tokens")
    for name in ("think", "answer"):
        for lo_b, hi_b in BUCKETS:
            if sum(r[f"n_{name}_{lo_b}"] for r in recs) > 0:
                slope([r[f"s_{name}_{lo_b}"] for r in recs], R, f"{name} tokens, step {lo_b}-{hi_b}")
    print("effect of the final-norm gain on the CAPPED RATE (negative = fewer capped):")
    slope([r["s_total"] for r in recs], capped, "all tokens")
    for lo_b, hi_b in BUCKETS:
        slope([r[f"s_think_{lo_b}"] for r in recs], capped, f"think tokens, step {lo_b}-{hi_b}")
    for family in sorted(set(fam)):
        m = torch.tensor([f == family for f in fam])
        if int(m.sum()) >= 30:
            s = torch.tensor([r["s_total"] for r in recs], dtype=torch.float64)[m]
            prod = (R[m] - R[m].mean()) * s
            print(f"  family {family:7s} n={int(m.sum()):3d}: score slope per x1.05 step {100 * prod.mean() * 0.0488:+5.2f} +- {100 * prod.std() / int(m.sum()) ** 0.5 * 0.0488:4.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(sys.argv[2], sys.argv[3])
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("mode")
        ap.add_argument("--model", required=True)
        ap.add_argument("--jobs", required=True)
        ap.add_argument("--run", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--shard", default="0/1")
        ap.add_argument("--chunk", type=int, default=128)
        collect(ap.parse_args())
