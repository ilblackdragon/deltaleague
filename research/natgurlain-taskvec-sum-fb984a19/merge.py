"""Task-vector sum on a shared parent: theta = P + wa*(A - P) + wb*(B - P).

No training, no GPU, no randomness. Arithmetic in float64, stored in the parent's dtypes.
Usage:
  python merge.py PARENT.safetensors A.safetensors B.safetensors OUT.safetensors [--wa 1.0 --wb 1.0] [--report report.json]
"""
import argparse, hashlib, json
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256(path, bs=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(bs):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parent"); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("out")
    ap.add_argument("--wa", type=float, default=1.0)
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--report")
    args = ap.parse_args()

    out, rows = {}, []
    dot = na = nb = 0.0
    with safe_open(args.parent, "pt") as fp, safe_open(args.a, "pt") as fa, safe_open(args.b, "pt") as fb:
        keys = sorted(fp.keys())
        assert keys == sorted(fa.keys()) == sorted(fb.keys()), "tensor names differ"
        for k in keys:
            p, a, b = fp.get_tensor(k), fa.get_tensor(k), fb.get_tensor(k)
            assert p.shape == a.shape == b.shape and p.dtype == a.dtype == b.dtype, k
            if not p.dtype.is_floating_point:
                assert torch.equal(p, a) and torch.equal(p, b), f"non-float tensor differs: {k}"
                out[k] = p
                continue
            p64 = p.double()
            da, db = a.double() - p64, b.double() - p64
            ca, cb = int((da != 0).sum()), int((db != 0).sum())
            if ca == 0 and cb == 0:
                out[k] = p
                continue
            m = (p64 + args.wa * da + args.wb * db).to(p.dtype)
            assert torch.isfinite(m).all(), k
            out[k] = m.contiguous()
            d = float((da * db).sum()); dot += d
            na += float((da * da).sum()); nb += float((db * db).sum())
            rows.append({"name": k, "numel": p.numel(), "changed_a": ca, "changed_b": cb,
                         "changed_both": int(((da != 0) & (db != 0)).sum()),
                         "changed_out": int((m != p).sum()),
                         "rel_a": float(da.norm() / p64.norm()), "rel_b": float(db.norm() / p64.norm()),
                         "rel_out": float((m.double() - p64).norm() / p64.norm())})
    save_file(out, args.out, metadata={"format": "pt"})
    rep = {"wa": args.wa, "wb": args.wb, "tensors_total": len(keys), "tensors_changed": sum(r["changed_out"] > 0 for r in rows),
           "tensors_changed_a": sum(r["changed_a"] > 0 for r in rows), "tensors_changed_b": sum(r["changed_b"] > 0 for r in rows),
           "elements_changed_out": sum(r["changed_out"] for r in rows),
           "delta_norm_a": na ** 0.5, "delta_norm_b": nb ** 0.5,
           "cosine_a_b": dot / ((na * nb) ** 0.5) if na and nb else None,
           "sha256": {"parent": sha256(args.parent), "a": sha256(args.a), "b": sha256(args.b), "out": sha256(args.out)},
           "per_tensor": rows}
    print(json.dumps({k: v for k, v in rep.items() if k != "per_tensor"}, indent=1))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
