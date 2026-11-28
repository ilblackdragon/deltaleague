"""Apply the isolated final-norm gain to the merged warm-start checkpoint."""
import argparse
from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    weights = "model.safetensors"
    norm = "model.language_model.norm.weight"
    with safe_open(args.source / weights, "pt") as file:
        metadata = file.metadata()
    tensors = load_file(args.source / weights)
    old = tensors[norm]
    tensors[norm] = (1.4 * (1.0 + old.float()) - 1.0).to(old.dtype)
    shutil.copytree(args.source, args.destination, ignore=shutil.ignore_patterns(weights))
    save_file(tensors, args.destination / weights, metadata=metadata)


if __name__ == "__main__":
    main()
