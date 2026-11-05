"""Prepare the recorded experiment without downloading hidden data or submitting."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, required=True, help="Pinned qwen35-particle-sliders checkout")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, code = args.output.resolve(), args.code.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source = root / "gsm8k-train.jsonl"
    url = "https://raw.githubusercontent.com/openai/grade-school-math/b0bb162abedc65e1fdd8e93ed090fd7598ee68bc/grade_school_math/data/train.jsonl"
    if not source.exists():
        with urlopen(url, timeout=120) as response:
            source.write_bytes(response.read())
    if hashlib.sha256(source.read_bytes()).hexdigest() != "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465":
        raise ValueError("Training source bytes differ from the recorded experiment")
    from qwen35_sliders.data import prepare_gsm
    prepare_gsm(source, root / "data.json")
    search = json.loads((code / "configs/search.json").read_text())
    search["base"].update(workspace=str(root / "experiment"), data=str(root / "data.json"))
    search["base"]["upstream"] = str(root / "particle-sliders-pg82")
    search["output"] = str(root / "runs/particle-search-v1")
    lora = json.loads((code / "configs/lora.json").read_text())
    lora.update(workspace=str(root / "experiment"), data=str(root / "data.json"), output=str(root / "runs/lora-v1"))
    for name, value in (("search", search), ("lora", lora)):
        (root / (name + ".json")).write_text(json.dumps(value, indent=2)+"\n")
    print("Prepared", root)
    print("Initialize the immutable parent, then run the bounded training commands:")
    print("delta competition init qwen35-08b-v1 --parent 0ea62f40-7cd8-4d98-9ebb-484192fa05eb --output", root / "experiment")
    print("qwen35-slider search", root / "search.json")
    print("qwen35-slider train-lora", root / "lora.json")


if __name__ == "__main__":
    main()
