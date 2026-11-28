"""Prepare and check Phase B rehearsal prompts, without evaluation-set data.

Sources: Google's IFEval supplies the idea of checking explicit constraints, not
its prompts. MBPP uses the full/train shard (CC BY 4.0) and Delta's native prompt
and WASI grader. Logic and tool-use tasks are generated from original templates.
"""

import argparse
import atexit
import itertools
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import threading


MBPP_REPO = "google-research-datasets/mbpp"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
MBPP_TRAIN = "full/train-00000-of-00001.parquet"
SEED = 20261126

TOPICS = (
    "planning a neighbourhood picnic", "organizing a shared kitchen",
    "keeping a balcony herb garden", "packing for a cycling trip",
    "starting a book club", "working on a school mural",
    "storing craft supplies", "making a museum visit accessible",
    "planning a low-cost dinner", "choosing a beginner's telescope",
    "arranging a community coat swap", "finding a quiet spot to read",
    "caring for houseplants in winter", "setting up a family game night",
    "preparing a bicycle for spring", "setting up a reading corner",
    "teaching a friend a board game", "volunteering at a walking event",
    "setting up a study area", "preparing for a rainy hike",
    "hosting a seed exchange", "keeping a field notebook",
    "visiting a local library", "taking photographs on a nature walk",
    "tidying a desk for a new term", "sharing tools with neighbours",
    "preparing a loose-leaf tea tasting", "planning a ferry outing",
    "sharing leftovers after a potluck", "starting a lunchtime walking group",
    "painting a small room", "selecting a trail for a mixed-age group",
    "practising a new musical instrument", "choosing games for a long train ride",
    "setting up a used-book table", "storing fresh berries",
    "organizing a jigsaw puzzle evening", "getting ready for a guest speaker",
    "drafting a dog-walking rota", "packing instruments after a rehearsal",
    "warming up before a short run", "repairing a torn backpack strap",
    "choosing plants for a shady porch", "scheduling a family photo",
    "preparing a borrowed tent for return", "planning a picnic at the beach",
    "practising conversational French", "starting a regular journalling habit",
)
CONSTRAINTS = (
    ("If the reader has never done this before", "guide"),
    ("For a group of three people", "people"),
    ("For someone with only half an hour", "time"),
    ("For a person using only materials already at hand", "materials"),
)
SYNTHETIC_NAMES = (
    "Amara", "Benoit", "Celia", "Devin", "Esme", "Farah", "Gio", "Hana",
    "Idris", "Jun", "Kira", "Luz", "Mina", "Noor", "Oleg", "Pia",
)


def row(item_id, bench, family, question, answer, **fields):
    return dict(id=item_id, bench=bench, family=family, question=question,
                answer=answer, prompt_text=question, **fields)


def instructions():
    # A Cartesian product varies both content and audience, independently of format.
    for index, (topic, (audience, keyword)) in enumerate(itertools.product(TOPICS, CONSTRAINTS)):
        stem = f"{audience}, give practical advice about {topic}. "
        prompts = (
            (f"Write exactly four Markdown bullet points, each starting with '- '. "
             f"Use the standalone word '{keyword}' once. Include no heading or introduction.",
             {"format": "bullets", "count": 4, "keyword": keyword}),
            (f"Use the headings 'Before' and 'During', in that order, each on its own line "
             f"with no Markdown. Write one paragraph under each. Use the word '{keyword}' "
             f"at least once, and omit a concluding section.",
             {"format": "sections", "headings": ["Before", "During"], "keyword": keyword}),
            (f"Begin with the standalone line 'Quick plan' and end with the standalone line "
             f"'Ready to begin'. Between those lines write 45 to 85 words. Avoid the word '{keyword}'.",
             {"format": "bounded", "start": "Quick plan", "end": "Ready to begin", "forbidden": keyword}),
            ("Return a JSON object with exactly the keys 'first_step' and 'backup_plan', "
             "each mapped to a string with at least eight words. Include no Markdown or other text.",
             {"format": "json", "keys": ["first_step", "backup_plan"]}),
        )
        for format_index, (requirement, checker) in enumerate(prompts):
            yield row(f"original-instruction-{index}-{format_index}", "instructions", "instruction",
                      stem + requirement, checker)


def logic(rng):
    symbols = ("book", "chair", "frame", "lantern", "mug", "plant", "ring", "scarf", "vase")
    for index in range(270):
        if index % 3 == 0:
            first, second, third, fourth = rng.sample(SYNTHETIC_NAMES, 4)
            clues = [f"{second} arrived immediately before {third}.",
                     f"{fourth} arrived after {third}.", f"{first} arrived before {second}."]
            rng.shuffle(clues)
            question = (f"Four volunteers arrived one by one. {' '.join(clues)} "
                        "Which volunteer arrived third? Give just the name.")
            answer = third
        elif index % 3 == 1:
            left, centre, right = rng.sample(symbols, 3)
            names = sorted((left, centre, right))
            question = (f"A {names[0]}, a {names[1]}, and a {names[2]} are on three adjacent shelves. "
                        f"The {left} is above the {centre}, and the {right} is below the {left}. "
                        f"The {centre} is not on the bottom shelf. What is on the middle shelf? "
                        "Give just the object's name.")
            answer = centre
        else:
            boxes = rng.sample(("gold", "silver", "copper", "violet", "teal", "white", "brown", "black"), 3)
            named, other, unmentioned = boxes
            question = (f"Exactly one of the {', '.join(boxes)} boxes holds a coin. Exactly one of "
                        f"these statements is true: (1) The coin is in the {named} box. "
                        f"(2) The coin is not in the {other} box. Which box holds the coin? "
                        "Give only its colour.")
            answer = unmentioned
        yield row(f"original-logic-{index}", "logic", "logic", question, answer)


def function(name, description, **properties):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": list(properties), "additionalProperties": False,
    }}}


def tool_use(rng):
    string = {"type": "string"}
    integer = {"type": "integer"}
    boolean = {"type": "boolean"}
    tasks = (
        (function("lookup_order", "Look up an order by its reference.", reference=string, include_line_items=boolean),
         function("lookup_inventory", "Look up stock by item code.", item_code=string),
         "Check order {reference} and include its individual line items.",
         "Check order {reference} but do not include its individual line items."),
        (function("reserve_desks", "Reserve a number of desks in a named workspace.", workspace=string, seats=integer),
         function("list_desks", "List desks in a named workspace.", workspace=string),
         "Reserve {seats} desks for our team in the {workspace} workspace.", None),
        (function("find_books", "Find library books given a list of keywords.", keywords={"type": "array", "items": string}),
         function("find_events", "Find library events by keyword.", keyword=string),
         "Search the library catalogue for books by these two keywords: {keywords[0]} and {keywords[1]}.", None),
    )
    workspaces = ("Cedar", "Harbour", "Maple", "Birch", "Juniper", "Aspen", "Balsam", "Spruce", "Ash", "Willow")
    terms = ("astronomy", "cycling", "architecture", "folklore", "gardening", "ecology", "photography",
             "cooking", "geology", "painting", "rivers", "music", "crafts", "history", "science", "trains")
    for index in range(300):
        task_index = index % len(tasks)
        active, distractor, template, negative_template = tasks[task_index]
        if task_index == 0:
            include_line_items = bool(rng.randrange(2))
            args = {"reference": f"{rng.randrange(10000):05}", "include_line_items": include_line_items}
            if not include_line_items:
                template = negative_template
        elif task_index == 1:
            args = {"workspace": rng.choice(workspaces), "seats": rng.randrange(1, 21)}
        else:
            args = {"keywords": rng.sample(terms, 2)}
        tools = [active, distractor]
        rng.shuffle(tools)
        question = template.format(**args) + " Use the available tools. Return only the tool call."
        answer = {"name": active["function"]["name"], "arguments": args}
        yield row(f"original-tool-{index}", "tool_call", "logic", question, answer, tools=tools)


def mbpp():
    from delta_cli.public_eval.coding import prompt
    from huggingface_hub import hf_hub_download
    from pyarrow.parquet import read_table

    parquet = hf_hub_download(MBPP_REPO, MBPP_TRAIN, revision=MBPP_REVISION, repo_type="dataset")
    fields = ["task_id", "text", "test_list", "test_setup_code"]
    docs = read_table(parquet, columns=fields).to_pylist()
    if sorted(doc["task_id"] for doc in docs) != list(range(601, 975)):
        raise ValueError("Unexpected MBPP full/train task IDs")
    for doc in docs:
        item = row(f"mbpp-train-{doc['task_id']}", "mbpp", "instruction", doc["text"],
                   {"test_list": doc["test_list"], "test_setup_code": doc["test_setup_code"] or ""})
        item["prompt_text"] = prompt("mbpp", doc)
        yield item


def word_count(text):
    return len(re.findall(r"\b[\w’'-]+\b", text))


def word_occurrences(text, word):
    return len(re.findall(rf"\b{re.escape(word)}\b", text, re.IGNORECASE))


def follows_instructions(spec, final):
    lines = final.strip().splitlines()
    match spec["format"]:
        case "bullets":
            return (len(lines) == spec["count"] and all(re.fullmatch(r"- \S.*", line) for line in lines)
                    and word_occurrences(final, spec["keyword"]) == 1)
        case "sections":
            nonempty = [line for line in lines if line.strip()]
            return (len(nonempty) == 4 and nonempty[::2] == spec["headings"]
                    and all(word_count(line) >= 8 for line in nonempty[1::2])
                    and word_occurrences(final, spec["keyword"]) >= 1)
        case "bounded":
            return (len(lines) >= 3 and lines[0] == spec["start"] and lines[-1] == spec["end"]
                    and 45 <= word_count("\n".join(lines[1:-1])) <= 85
                    and word_occurrences(final, spec["forbidden"]) == 0)
        case "json":
            try:
                value = json.loads(final, object_pairs_hook=unique_object)
            except (ValueError, RecursionError):
                return False
            return (isinstance(value, dict) and set(value) == set(spec["keys"])
                    and all(isinstance(text, str) and word_count(text) >= 8 for text in value.values()))
    raise ValueError(f"Unknown instruction checker {spec['format']}")


def unique_object(pairs):
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("Duplicate JSON key")
    return obj


def valid_tool_call(item, final):
    # Qwen3.5's chat template uses XML-like call delimiters and freshly typed
    # argument values. A JSON-formatted call inside the same delimiters stays
    # compatible with the pinned Delta BFCL adapter.
    match = re.fullmatch(r"<tool_call>\s*(.*?)\s*</tool_call>", final, re.DOTALL)
    if not match:
        return False
    body = match[1]
    try:
        if body.startswith("{"):
            call = json.loads(body, object_pairs_hook=unique_object)
        else:
            function_match = re.fullmatch(r"<function=([\w.\-]+)>\s*(.*?)\s*</function>", body, re.DOTALL)
            if not function_match:
                return False
            name, body = function_match.groups()
            schemas = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in item["tools"]}
            schema = schemas.get(name, {})
            pairs = []
            end = 0
            for argument in re.finditer(r"<parameter=([\w.\-]+)>\s*(.*?)\s*</parameter>", body, re.DOTALL):
                key, value = argument.groups()
                if body[end:argument.start()].strip() or key not in schema:
                    return False
                pairs.append((key, value if schema[key]["type"] == "string" else json.loads(value)))
                end = argument.end()
            if body[end:].strip():
                return False
            call = {"name": name, "arguments": unique_object(pairs)}
        return json.dumps(call, sort_keys=True) == json.dumps(item["answer"], sort_keys=True)
    except (ValueError, RecursionError):
        return False


_worker = None
_worker_lock = threading.Lock()


def check_mbpp(doc, final):
    global _worker
    with _worker_lock:
        if _worker is None:
            python = os.environ.get("DELTA_REHEARSAL_PYTHON", ".venv-public/bin/python")
            env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
            env.setdefault("DELTA_WASI_ROOT", str(Path(".wasi").resolve()))
            _worker = subprocess.Popen([python, str(Path(__file__).resolve()), "--wasi-worker"],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
            atexit.register(close_worker)
        data = json.dumps({"doc": doc, "final": final})
        _worker.stdin.write(data + "\n")
        _worker.stdin.flush()
        outcome = _worker.stdout.readline()
        if not outcome:
            raise RuntimeError("MBPP WASI worker exited before returning a result")
        return json.loads(outcome)["passed"]


def close_worker():
    _worker.stdin.close()
    _worker.wait(timeout=20)
    _worker.stdout.close()


def wasi_worker():
    from delta_cli.public_eval.coding import extract_code, program
    from delta_cli.public_eval.wasi import sandbox

    if not sandbox("assert 1 + 1 == 2")["passed"]:
        raise RuntimeError("MBPP WASI sandbox preflight failed")
    for line in sys.stdin:
        e = json.loads(line)
        code = extract_code(e["final"])
        print(json.dumps(sandbox(program("mbpp", e["doc"], code))), flush=True)


def is_correct(item, final):
    final = final.strip()
    if not final:
        return False
    bench, answer = item["bench"], item["answer"]
    if bench == "instructions":
        return follows_instructions(answer, final)
    if bench == "logic":
        return final.casefold().removesuffix(".") == answer.casefold()
    if bench == "tool_call":
        return valid_tool_call(item, final)
    if bench == "mbpp":
        return check_mbpp(answer, final)
    raise ValueError(f"Unknown rehearsal benchmark {bench}")


def main():
    if sys.argv[1:] == ["--wasi-worker"]:
        wasi_worker()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    rng = random.Random(SEED)
    items = list(itertools.chain(instructions(), mbpp(), logic(rng), tool_use(rng)))
    seen = set()
    with args.out.open("x") as out:
        for item in items:
            key = item["bench"], item["prompt_text"]
            if key not in seen:
                print(json.dumps(item, ensure_ascii=False), file=out)
                seen.add(key)
    print(f"Wrote {len(seen)} unique rehearsal prompts")


if __name__ == "__main__":
    main()
