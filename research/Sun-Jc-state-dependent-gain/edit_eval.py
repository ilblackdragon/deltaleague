"""Exact effect of a weight edit on stored completions, at the level of the evaluation sampler.

Teacher-forces every selected completion through the parent and through the edited model
and compares, at every generated position, the distributions the sampler would draw from
(presence penalty on tokens already generated, top-k 20, top-p 0.95).

Because the edited model reads the same token path, its state at step t is exactly the
state it would have after sampling those t tokens itself. So the numbers are exact
conditional probabilities along each stored path:
  tv        total variation between the two sampling distributions = chance that a coupled
            sampler picks a different token at this step
  watched   probability of chosen tokens (default </think>, backslash, EOS) under both models

collect: python edit_eval.py collect --model FP32_DIR --edit EDIT.pt --jobs JOBS --run RUN --out OUT
                             [--only ok,runaway,overflow] [--shard K/P] [--max-rows N] [--watch IDS] [--thinking]
report:  python edit_eval.py report OUT [OUT ...]
"""
import argparse
import glob
import json
import os
import sys

import torch

DEFAULT_WATCH = "248069,59,248046"  # </think>, backslash, <|im_end|>


def sampler_tv(ids_b, q_b, ids_e, q_e):
    match = ids_b[:, :, None] == ids_e[:, None, :]
    e_on_b = (match * q_e[:, None, :]).sum(-1)
    unmatched_e = q_e * (~match.any(1))
    return 0.5 * ((q_b - e_on_b).abs().sum(-1) + unmatched_e.sum(-1))


def collect(args):
    import edits as edits_lib
    from engine import EOS_IDS, THINK_CLOSE, Model, Sampler
    from scorelib import load_rows, outcome

    k, p = (int(x) for x in args.shard.split("/"))
    os.makedirs(args.out, exist_ok=True)
    base = Model(args.model).install_fast_paths()
    edited = Model(args.model).install_fast_paths()
    edits_lib.apply_to_model(edited, edits_lib.load(args.edit))
    sampler = Sampler()
    watch = [int(x) for x in args.watch.split(",")]
    jobs = {j["id"]: j for j in map(json.loads, open(args.jobs))}
    rows = sorted(load_rows(args.run), key=lambda r: (r["id"], r["sample"]))
    only = set(args.only.split(",")) if args.only else None
    rows = [r for r in rows if only is None or outcome(r) in only]
    if args.max_rows:
        rows = rows[:args.max_rows]
    rows = [r for i, r in enumerate(rows) if i % p == k]
    meta, tv_all, pw_b, pw_e, pt_b, pt_e, top_b, top_e, kept = [], [], [], [], [], [], [], [], []
    with torch.inference_mode():
        for ri, r in enumerate(rows):
            prompt = base.encode_prompt(jobs[r["id"]]["prompt"], thinking=not args.no_thinking)
            toks = list(r["tokens"])
            n = len(toks)
            targets = toks + ([EOS_IDS[0]] if r["finish"] == "stop" else [])  # step n predicts EOS when the run stopped
            ids = torch.tensor([prompt + toks], dtype=torch.long)
            n_prompt = len(prompt)
            first_seen = torch.full((base.vocab,), 1 << 30, dtype=torch.long)
            for i, t in enumerate(toks):
                if first_seen[t] > i:
                    first_seen[t] = i
            first_close = toks.index(THINK_CLOSE) if THINK_CLOSE in toks else -1
            last_step = len(targets) - 1
            if args.thinking and first_close >= 0:
                last_step = min(last_step, first_close)  # only the thinking phase (incl. the closing step)
            last_pos = n_prompt - 1 + last_step
            past_b, past_e = base.new_cache(reserve=ids.shape[1] + 8), edited.new_cache(reserve=ids.shape[1] + 8)
            for a in range(0, last_pos + 1, args.chunk):
                b = min(last_pos + 1, a + args.chunk)
                hb = base.text(input_ids=ids[:, a:b], past_key_values=past_b, use_cache=True).last_hidden_state[0]
                he = edited.text(input_ids=ids[:, a:b], past_key_values=past_e, use_cache=True).last_hidden_state[0]
                lo = max(a, n_prompt - 1)
                if lo >= b:
                    continue
                steps = torch.arange(lo - (n_prompt - 1), b - (n_prompt - 1))
                seen = (first_seen[None, :] < steps[:, None]).float()
                ids_b, q_b, _ = sampler.candidates(base.head(hb[lo - a:]), seen)
                ids_e, q_e, _ = sampler.candidates(edited.head(he[lo - a:]), seen)
                tv_all.append(sampler_tv(ids_b, q_b, ids_e, q_e))
                tgt = torch.tensor([targets[s] for s in steps.tolist()])
                pw_b.append(torch.stack([(q_b * (ids_b == w)).sum(-1) for w in watch], 1))
                pw_e.append(torch.stack([(q_e * (ids_e == w)).sum(-1) for w in watch], 1))
                pt_b.append((q_b * (ids_b == tgt[:, None])).sum(-1))
                pt_e.append((q_e * (ids_e == tgt[:, None])).sum(-1))
                top_b.append(torch.gather(ids_b, 1, q_b.argmax(-1, keepdim=True)).squeeze(1))
                top_e.append(torch.gather(ids_e, 1, q_e.argmax(-1, keepdim=True)).squeeze(1))
                for s in steps.tolist():
                    meta.append((len(kept), s, -1 if first_close < 0 or s <= first_close else s - first_close, targets[s]))
            kept.append({"id": r["id"], "sample": r["sample"], "outcome": outcome(r), "family": jobs[r["id"]]["family"], "n": n})
            if (ri + 1) % 5 == 0 or ri + 1 == len(rows):
                print(f"[edit_eval {k}/{p}] {ri + 1}/{len(rows)} rows", file=sys.stderr, flush=True)
    torch.save({"meta": torch.tensor(meta, dtype=torch.int32), "tv": torch.cat(tv_all), "watch": watch,
                "p_watch_base": torch.cat(pw_b), "p_watch_edit": torch.cat(pw_e), "p_tok_base": torch.cat(pt_b), "p_tok_edit": torch.cat(pt_e),
                "top_base": torch.cat(top_b), "top_edit": torch.cat(top_e), "rows": kept}, os.path.join(args.out, f"eval-{k}.pt"))


def load_parts(dirs):
    out, rows = {}, []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(os.path.expanduser(d), "eval-*.pt"))):
            part = torch.load(f)
            part["meta"] = part["meta"].clone()
            part["meta"][:, 0] += len(rows)
            rows.extend(part["rows"])
            for key, v in part.items():
                if key not in ("rows", "watch"):
                    out.setdefault(key, []).append(v)
            watch = part["watch"]
    return {k: torch.cat(v) for k, v in out.items()}, rows, watch


def report(dirs):
    d, rows, watch = load_parts(dirs)
    meta = d["meta"]
    row, step, since = meta[:, 0].long(), meta[:, 1].long(), meta[:, 2].long()
    thinking = since < 0
    tv = d["tv"].double()
    capped = torch.tensor([r["outcome"] in ("runaway", "overflow") for r in rows])[row]
    print(f"{len(rows)} completions, {len(tv):,} positions; watched tokens {watch}")
    close_col = watch.index(248069) if 248069 in watch else None

    def block(name, sel):
        if int(sel.sum()) == 0:
            return
        t = tv[sel]
        line = (f"  {name:34s} n={int(sel.sum()):8,d}  TV mean {t.mean():.2e}  p99 {torch.quantile(t, 0.99) if len(t) < 2_000_000 else t.sort().values[int(0.99 * len(t))]:.2e}"
                f"  max {t.max():.3f}  top-1 changed {100 * (d['top_base'][sel] != d['top_edit'][sel]).double().mean():.3f}%")
        if close_col is not None:
            line += f"  P(</think>) {d['p_watch_base'][sel, close_col].double().mean():.4f} -> {d['p_watch_edit'][sel, close_col].double().mean():.4f}"
        print(line)

    print("normal completions (must stay unchanged):")
    for a, b in [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000), (5000, 6000), (6000, 7000), (7000, 8200)]:
        block(f"thinking, step {a}-{b}", ~capped & thinking & (step >= a) & (step < b))
    block("answering", ~capped & ~thinking)
    print("capped completions:")
    for a, b in [(0, 2000), (2000, 4000), (4000, 5000), (5000, 6000), (6000, 6500), (6500, 7000), (7000, 7500), (7500, 8200)]:
        block(f"thinking, step {a}-{b}", capped & thinking & (step >= a) & (step < b))
        block(f"answering, step {a}-{b}", capped & ~thinking & (step >= a) & (step < b))
    # Chance that a coupled sampler leaves the stored path at all, per completion.
    n_rows = len(rows)
    log_stay = torch.zeros(n_rows, dtype=torch.float64).index_add_(0, row, torch.log1p(-tv.clamp(max=1 - 1e-12)))
    leave = 1 - log_stay.exp()
    is_capped = torch.tensor([r["outcome"] in ("runaway", "overflow") for r in rows])
    if int((~is_capped).sum()):
        v = leave[~is_capped]
        print(f"normal completions: P(sampled path differs anywhere) mean {v.mean():.4f}, median {v.median():.4f}, max {v.max():.4f}; "
              f"{int((v > 0.05).sum())} of {len(v)} above 5%")
    if close_col is not None and int(is_capped.sum()):
        # Exit time under the edited model along capped paths: first step where the survival of "no </think>" falls below 1/2.
        pc = d["p_watch_edit"][:, close_col].double().clamp(max=1 - 1e-12)
        exits = []
        for r in is_capped.nonzero().flatten().tolist():
            sel = (row == r) & thinking
            s, h = step[sel], pc[sel]
            order = s.argsort()
            surv = torch.exp(torch.cumsum(torch.log1p(-h[order]), 0))
            below = (surv < 0.5).nonzero()
            exits.append(int(s[order][below[0, 0]]) if len(below) else -1)
        done = sorted(e for e in exits if e >= 0)
        print(f"capped completions: edited model closes thinking (survival < 1/2) in {len(done)} of {len(exits)}; "
              f"exit steps: {done}")


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(sys.argv[2:])
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("mode")
        ap.add_argument("--model", required=True)
        ap.add_argument("--edit", required=True)
        ap.add_argument("--jobs", required=True)
        ap.add_argument("--run", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--only", default="")
        ap.add_argument("--shard", default="0/1")
        ap.add_argument("--max-rows", type=int, default=0)
        ap.add_argument("--watch", default=DEFAULT_WATCH)
        ap.add_argument("--chunk", type=int, default=128)
        ap.add_argument("--thinking", action="store_true", help="stop each completion at its first </think>")
        ap.add_argument("--no-thinking", action="store_true", help="render prompts with thinking disabled")
        collect(ap.parse_args())
