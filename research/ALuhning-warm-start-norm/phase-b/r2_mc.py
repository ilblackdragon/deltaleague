"""Render pinned lm-eval choices from training files for thinking rollouts.

Source templates: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
Native likelihood prompts need their implicit alternatives made visible for sampling.
The original continuations remain the requested answers. Canonical letters are
used internally for scoring and comparison across agreement cuts.
"""

import argparse
from collections import Counter
import hashlib
import heapq
import json
from pathlib import Path
import re
from string import ascii_uppercase
import unicodedata

SHARD = "train-00000-of-00001.parquet"
# Only these explicit training files may be opened. MMLU uses auxiliary_train.
SOURCES = {
    "arc_challenge": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", f"ARC-Challenge/{SHARD}"),
    "arc_easy": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", f"ARC-Easy/{SHARD}"),
    "commonsense_qa": ("tau/commonsense_qa", "94630fe30dad47192a8546eb75f094926d47e155", f"data/{SHARD}"),
    "openbookqa": ("allenai/openbookqa", "388097ea7776314e93a529163e0fea805b8a6454", f"main/{SHARD}"),
    "piqa": ("baber/piqa", "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1", "piqa_train.parquet"),
    "winogrande": ("allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5", f"winogrande_xl/{SHARD}"),
    "hellaswag": ("Rowan/hellaswag", "218ec52e09a7e7462a5400043bb9a69a41d06b76", f"data/{SHARD}"),
    "boolq": ("aps/super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646", f"boolq/{SHARD}"),
    "mmlu": ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", f"auxiliary_train/{SHARD}"),
}


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _answer_text(text):
    text = normalize(text).strip()
    text = re.sub(r"^(?:(?:final\s+)?answer\s*:\s*)", "", text)
    box = re.fullmatch(r"\\boxed\{(.*)\}\s*\.?", text)
    if box:
        text = box[1]
    text = text.rstrip(".!?").strip().strip("\"“”'‘’*`")
    return text.strip().rstrip(".!?").strip()


def choice(item, final):
    """Return a canonical option letter only for an unambiguous native answer."""
    options = item["native_answers"]
    answer = _answer_text(final)
    if item["answer_format"] == "letter":
        answer = answer.strip("()[]").rstrip(")").strip()
    matches = [ascii_uppercase[i] for i, option in enumerate(options) if _answer_text(option) == answer]
    return matches[0] if len(matches) == 1 else None


def _hellaswag(text):
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def _options(context, options, directive):
    displayed = "\n".join(f"- {option}" for option in options)
    return f"{directive}\n\nOptions:\n{displayed}\n\n{context}"


def render(bench, row):
    """Return source question, generative user prompt, native answers and gold index."""
    if bench in ("arc_challenge", "arc_easy", "commonsense_qa", "openbookqa"):
        question = row["question_stem"] if bench == "openbookqa" else row["question"]
        options = row["choices"]["text"]
        gold = row["choices"]["label"].index(row["answerKey"].strip())
        if bench != "commonsense_qa":
            context = question if bench == "openbookqa" else f"Question: {question}\nAnswer:"
            prompt = _options(context, options, "Choose the correct answer. Give the exact option text.")
            return question, prompt, options, gold, "text"
        prefix = "Question: "
    elif bench == "mmlu":
        row = row["train"]
        question, options, gold = row["question"], row["choices"], row["answer"]
        prefix = ""
    elif bench == "piqa":
        question = row["goal"]
        options = [row["sol1"], row["sol2"]]
        context = f"Question: {question}\nAnswer:"
        prompt = _options(context, options, "Choose the correct answer. Give the exact option text.")
        return question, prompt, options, row["label"], "text"
    elif bench == "boolq":
        question = row["question"]
        context = f"{row['passage']}\nQuestion: {question}?\nAnswer:"
        prompt = f"Answer the question with either yes or no.\n\n{context}"
        return question, prompt, ["no", "yes"], row["label"], "text"
    elif bench == "hellaswag":
        question = _hellaswag(f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}")
        options = [_hellaswag(ending) for ending in row["endings"]]
        prompt = _options(question, options, "Choose the most plausible continuation. Give the exact continuation.")
        return question, prompt, options, int(row["label"]), "text"
    elif bench == "winogrande":
        question = row["sentence"]
        beginning = question.split("_", 1)[0]
        options = [beginning + row["option1"], beginning + row["option2"]]
        prompt = _options(
            question,
            options,
            "Choose the prefix that fills the blank in the sentence. Give the exact prefix.",
        )
        return question, prompt, options, int(row["answer"]) - 1, "text"
    else:
        raise ValueError(f"Unknown task {bench}")

    letters = list(ascii_uppercase[: len(options)])
    displayed = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options, strict=True))
    prompt = f"Choose the correct option. Give its letter.\n\n{prefix}{question.strip()}\n{displayed}\nAnswer:"
    return question, prompt, letters, gold, "letter"


def rank(seed, bench, index):
    return hashlib.sha256(f"{seed}:{bench}:{index}".encode()).digest()


def shortlist(bench, seed, size):
    from huggingface_hub import hf_hub_download
    from pyarrow.parquet import ParquetFile

    repo, revision, filename = SOURCES[bench]
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    parquet = ParquetFile(path)
    docs = (row for batch in parquet.iter_batches(batch_size=4096) for row in batch.to_pylist())
    rows = heapq.nsmallest(size, enumerate(docs), key=lambda item: rank(seed, bench, item[0]))
    for index, row in rows:
        question, prompt, answers, gold, answer_format = render(bench, row)
        if not 0 <= gold < len(answers):
            raise ValueError(f"Invalid source label for {bench} row {index}")
        parsed = [_answer_text(option) for option in answers]
        if not all(parsed) or len(set(parsed)) != len(parsed):
            continue
        split = "auxiliary_train" if bench == "mmlu" else "train"
        yield {
            "id": f"{bench}-{split}-{index}",
            "bench": bench,
            "family": "mc",
            "question": question,
            "prompt_text": prompt,
            "answer": ascii_uppercase[gold],
            "answer_format": answer_format,
            "native_answers": answers,
            "source_split": split,
        }


def select(candidates, counts, seen):
    selected = []
    for bench, count in counts.items():
        picked = 0
        for item in candidates[bench]:
            if picked == count:
                break
            # Exclude repeated stems across partitions, including across sources.
            stem = re.sub(r"[^\w]+", " ", normalize(item["question"])).strip()
            if stem in seen:
                continue
            seen.add(stem)
            selected.append(item)
            picked += 1
        if picked != count:
            raise ValueError(f"{bench}: selected {picked} of {count} distinct items")
    return selected


def write(path, rows):
    with path.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("holdout", type=Path)
    parser.add_argument("--per-family", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args()
    if args.per_family < 2:
        parser.error("--per-family must be at least 2")
    if args.train.resolve() == args.holdout.resolve() or args.train.exists() or args.holdout.exists():
        parser.error("use distinct, new paths for train and holdout")
    held_counts = dict.fromkeys(SOURCES, 8)
    held_counts.update(arc_challenge=4, arc_easy=4)
    train_counts = dict.fromkeys(SOURCES, args.per_family)
    train_counts.update(arc_challenge=args.per_family // 2, arc_easy=(args.per_family + 1) // 2)
    candidates = {
        bench: list(shortlist(bench, args.seed, max(128, (train_counts[bench] + held_counts[bench]) * 4)))
        for bench in SOURCES
    }
    seen = set()
    held = select(candidates, held_counts, seen)
    train = select(candidates, train_counts, seen)
    write(args.holdout, held)
    write(args.train, train)
    print(json.dumps({
        "seed": args.seed,
        "train": dict(Counter(item["bench"] for item in train)),
        "holdout": dict(Counter(item["bench"] for item in held)),
        "sources": SOURCES,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
