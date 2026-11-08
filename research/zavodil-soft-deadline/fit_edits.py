"""Fit rank-1 edits for layer 23 down_proj from probe_states.py output.

usage: fit_edits.py PROBE.npz MODEL_DIR OUT_DIR

Finds in the down_proj input space (3584):
  u_phase: direction separating answer phase from thinking (mean difference)
  u_pos:   direction whose projection grows with position inside thinking (least squares)
And in residual space (1024):
  v_boxed: raises the logit of "\\boxed" through the tied embedding and final norm
  v_close: raises the logit of </think>
Then calibrates, by a forward pass on two traces, how much each unit of delta moves
the target logit, so build scripts can ask for a logit shift instead of a raw scale.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

THINK_CLOSE = 248069
LAYER = 23


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    positives = labels == 1
    return (ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) / (positives.sum() * (~positives).sum())


def main():
    probe, model_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(probe, allow_pickle=True)
    a, h, pos, phase, group, trace = d["a"].astype(np.float32), d["h"].astype(np.float32), d["pos"], d["phase"], d["group"], d["trace"]
    report = {}

    # Phase direction in down_proj-input space.
    think, answer = a[phase == 0], a[phase == 1]
    u_phase = answer.mean(0) - think.mean(0)
    u_phase /= np.linalg.norm(u_phase)
    proj = a @ u_phase
    report["phase_auc_a"] = float(auc(proj, phase))
    report["phase_proj_mean"] = {"think": float(proj[phase == 0].mean()), "answer": float(proj[phase == 1].mean()), "think_std": float(proj[phase == 0].std())}
    hp = (h.mean(0) * 0 + h[phase == 1].mean(0) - h[phase == 0].mean(0))
    report["phase_auc_h"] = float(auc(h @ hp, phase))

    # Position direction inside thinking, fitted across traces, validated by held-out traces.
    mask = phase == 0
    traces = np.unique(trace[mask])
    held = traces[::3]
    fit_mask = mask & ~np.isin(trace, held)
    val_mask = mask & np.isin(trace, held)
    X = np.hstack([a[fit_mask], np.ones((fit_mask.sum(), 1), np.float32)])
    y = pos[fit_mask].astype(np.float32)
    coef = np.linalg.lstsq(X, y, rcond=1e-3)[0]
    u_pos = coef[:-1]
    pred_val = a[val_mask] @ u_pos + coef[-1]
    report["pos_r_heldout_a"] = float(np.corrcoef(pred_val, pos[val_mask])[0, 1])
    corr = [abs(np.corrcoef(a[mask][:, i], pos[mask])[0, 1]) for i in range(a.shape[1])]
    report["pos_best_single_dim_a"] = {"dim": int(np.argmax(corr)), "r": float(max(corr))}
    corr_h = [abs(np.corrcoef(h[mask][:, i], pos[mask])[0, 1]) for i in range(h.shape[1])]
    report["pos_best_single_dim_h"] = {"dim": int(np.argmax(corr_h)), "r": float(max(corr_h))}
    u_pos_unit = u_pos / np.linalg.norm(u_pos)
    report["pos_proj_by_bucket"] = {}
    pr = a[mask] @ u_pos_unit
    for lo, hi in ((0, 500), (500, 1500), (1500, 3000), (3000, 5000), (5000, 8200)):
        sel = (pos[mask] >= lo) & (pos[mask] < hi)
        if sel.any():
            report["pos_proj_by_bucket"][f"{lo}-{hi}"] = float(pr[sel].mean())
    report["pos_phase_leak"] = {"answer_mean_proj": float((a[phase == 1] @ u_pos_unit).mean())}
    report["pos_proj_std_within_buckets"] = float(np.mean([pr[(pos[mask] >= lo) & (pos[mask] < hi)].std() for lo, hi in ((0, 500), (500, 1500), (1500, 3000), (3000, 5000)) if ((pos[mask] >= lo) & (pos[mask] < hi)).any()]))
    strict_answer = (phase == 1) & (group == "strict")
    overflow_answer = (phase == 1) & (group == "overflow")
    report["phase_proj_answer_by_group"] = {"strict": float(proj[strict_answer].mean()), "overflow": float(proj[overflow_answer].mean())}

    # Residual-space targets through the tied embedding and final norm.
    with safe_open(model_dir / "model.safetensors", "pt") as f:
        E = f.get_tensor("model.language_model.embed_tokens.weight").float()
        w = f.get_tensor("model.language_model.norm.weight").float()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    pieces = tok.encode("\\boxed{", add_special_tokens=False)
    boxed_id = next(t for t in pieces if "boxed" in tok.decode([t]))
    gain = 1 + w
    v_boxed = (E[boxed_id] * gain).numpy()
    v_close = (E[THINK_CLOSE] * gain).numpy()
    v_boxed /= np.linalg.norm(v_boxed)
    v_close /= np.linalg.norm(v_close)
    report["boxed_token"] = tok.decode([boxed_id])

    # Overflow-answer direction: answer phase of overflowing traces against everything else.
    ovf = (phase == 1) & (group == "overflow")
    u_ovf = a[ovf].mean(0) - a[~ovf].mean(0)
    u_ovf /= np.linalg.norm(u_ovf)
    po = a @ u_ovf
    report["ovf_auc"] = float(auc(po, ovf.astype(int)))
    report["ovf_proj"] = {"overflow_answer": float(po[ovf].mean()), "strict_answer": float(po[(phase == 1) & (group == "strict")].mean()), "think": float(po[phase == 0].mean()), "think_std": float(po[phase == 0].std())}
    # Whitened boxed direction: remove the mean embedding and the top principal components,
    # so the push lands on "boxed" rather than on LaTeX tokens in general.
    centered = E - E.mean(0, keepdim=True)
    sample = centered[torch.randperm(centered.shape[0])[:40000]]
    _, _, Vt = torch.linalg.svd(sample, full_matrices=False)
    top = Vt[:32]
    target = centered[boxed_id] - top.T @ (top @ centered[boxed_id])
    v_boxed_w = (target * gain).numpy()
    v_boxed_w /= np.linalg.norm(v_boxed_w)
    np.savez(out_dir / "directions.npz", u_phase=u_phase, u_pos=u_pos_unit, v_boxed=v_boxed, v_close=v_close, u_ovf=u_ovf, v_boxed_w=v_boxed_w,
             pos_intercept=coef[-1], pos_scale=np.linalg.norm(u_pos))
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
