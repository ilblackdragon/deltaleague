#!/usr/bin/env python3
"""Asks a hosted model every prompt of an item file and keeps the letter it gave. Standard library only.

    python ask_models.py ITEMS.json --endpoint https://cloud-api.near.ai/v1 --api-key-env NEAR_AI_API_KEY \\
        --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --extra '{"chat_template_kwargs": {"thinking": false}}' \\
        --header "x-no-aliasing: true" --out solvers.json

This is the solver gate's input. ITEMS.json is a file in the set's shape: the set itself, or every candidate
before the gate (`python build_set.py DIR --candidates candidates.json`). Each prompt goes as one user
message at temperature 0, with --extra merged into the request and a cap of --max-tokens output tokens.
The answer is read as the `choice` scorer reads it, and stored under the model's name and the sha256 of
the prompt, so a prompt is asked once and a changed prompt is asked again. Nothing but the letter, the
finish reason and the answer's length is kept.

The API key is read from the environment variable named by --api-key-env and is never printed or stored.
The endpoint must be https (plain http only for a server on this machine), and a redirect is never
followed, so the key goes nowhere but the endpoint named. A request that fails (after four tries on 429
and 5xx), or a reply that is not a chat completion with text, is not an answer: nothing is stored for it,
it is counted, and the exit code is 1.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_set import read_choice  # noqa: E402  (the same reader the builder documents)

NOTE = ("Answers of hosted models to this build's candidate prompts, keyed by model and by sha256(prompt). "
        "One user message, temperature 0, a cap on output tokens, and the request extras named in the README. "
        "`letter` is what the choice scorer reads (null when it reads none), `finish` is the API's finish reason "
        "and `chars` the length of the answer.")


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dumps(store):
    """One answer per line, models and prompts in sorted order, so two runs that agree give the same bytes."""
    models = []
    for model in sorted(store["models"]):
        rows = ",\n".join(f"{json.dumps(k)}:{json.dumps(v, separators=(',', ':'))}" for k, v in sorted(store["models"][model].items()))
        models.append(f"{json.dumps(model)}:{{\n{rows}\n}}")
    return f'{{"note":{json.dumps(store.get("note") or NOTE, ensure_ascii=False)},"models":{{\n' + ",\n".join(models) + "\n}}\n"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """An API answers where it is asked. Following a redirect would send the key to wherever it points."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


OPENER = urllib.request.build_opener(NoRedirect)
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def check_endpoint(endpoint):
    """HTTPS only, except for a server on this machine."""
    parts = urllib.parse.urlsplit(endpoint)
    if parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in LOOPBACK):
        return endpoint
    raise ValueError(f"{endpoint}: use https (plain http only for a server on this machine)")


def reading(reply):
    """The stored row for a reply, or {"error": ...} when the reply is not a chat completion."""
    if not isinstance(reply, dict) or "error" in reply or not isinstance(reply.get("choices"), list) or not reply["choices"]:
        return {"error": "the reply is not a chat completion: " + json.dumps(reply)[:100]}
    choice = reply["choices"][0] if isinstance(reply["choices"][0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else None
    if message is None:
        return {"error": "the reply holds no message"}
    content = message.get("content")
    if content is None and choice.get("finish_reason"):
        # a model that spent its whole budget thinking: an answer with no text, so no letter
        return {"letter": None, "finish": choice.get("finish_reason"), "chars": 0}
    if not isinstance(content, str):
        return {"error": "the reply holds no text"}
    return {"letter": read_choice(content), "finish": choice.get("finish_reason"), "chars": len(content)}


def ask(url, headers, body, tries=4, wait=5.0, timeout=180):
    """One chat completion. Returns the stored row, or {"error": ...} when no answer came back."""
    data = json.dumps(body).encode("utf-8")
    last = "no attempt"
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, data=data, headers={"content-type": "application/json", **headers})
            with OPENER.open(request, timeout=timeout) as response:
                return reading(json.loads(response.read()))
        except urllib.error.HTTPError as error:
            if 300 <= error.code < 400:
                return {"error": f"HTTP {error.code}: a redirect, not followed"}
            if error.code != 429 and error.code < 500:
                return {"error": f"HTTP {error.code}"}
            last = f"HTTP {error.code}"
        except (OSError, ValueError) as error:
            last = str(error)[:120]
        if attempt + 1 < tries:
            time.sleep(wait * (attempt + 1))
    return {"error": last}


def run(items, store, *, model, endpoint, headers, extra, max_tokens, limit=None, redo=False, unread=False, workers=4,
        save=lambda: None, wait=5.0):
    """Ask what is not answered yet. Returns (asked, errors)."""
    answers = store.setdefault("models", {}).setdefault(model, {})
    prompts = {}
    for rows in items.values():
        for row in rows[:limit] if limit else rows:
            prompts[sha256(row["prompt"])] = row["prompt"]
    todo = [k for k in prompts if redo or k not in answers or (unread and answers[k].get("letter") is None)]
    url = check_endpoint(endpoint).rstrip("/") + "/chat/completions"
    errors = 0

    def one(key):
        body = {"model": model, "messages": [{"role": "user", "content": prompts[key]}], "temperature": 0, **extra,
                "max_tokens": max_tokens, "stream": False}
        return key, ask(url, headers, body, wait=wait)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (key, row) in enumerate(pool.map(one, todo), 1):
            if "error" in row:
                errors += 1
                print(f"no answer for {key[:12]}: {row['error']}", file=sys.stderr)
            else:
                answers[key] = row
            if done % 25 == 0:
                save()
                print(f"{done} of {len(todo)} asked", file=sys.stderr)
    save()
    return len(todo), errors


def table(items, answers, limit=None):
    out = {}
    for family, rows in items.items():
        rows = rows[:limit] if limit else rows
        got = [answers.get(sha256(row["prompt"])) for row in rows]
        out[family] = {"items": len(rows), "right": sum(1 for a, row in zip(got, rows) if a and a.get("letter") == row.get("answer")),
                       "no_letter": sum(1 for a in got if a and a.get("letter") is None), "not_asked": sum(1 for a in got if not a)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ask a hosted model every prompt of an item file; keep the letter it gave.")
    ap.add_argument("items", help="a file in the set's shape: {family: [{id, prompt, answer, ...}]}")
    ap.add_argument("--endpoint", required=True, help="an OpenAI-compatible base URL, for example https://cloud-api.near.ai/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", help="name of the environment variable that holds the API key")
    ap.add_argument("--extra", default="{}", help="JSON merged into each request, for example a thinking switch")
    ap.add_argument("--header", action="append", default=[], help="an extra request header, as 'name: value'; repeatable")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", default="solvers.json")
    ap.add_argument("--limit", type=int, help="only the first N prompts of each family")
    ap.add_argument("--redo", action="store_true", help="ask again even where an answer is stored")
    ap.add_argument("--unread", action="store_true", help="ask again where no letter could be read")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    items = json.loads(Path(a.items).read_text(encoding="utf-8"))
    out = Path(a.out)
    store = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"note": NOTE, "models": {}}
    headers = {}
    for line in a.header:
        name, _, value = line.partition(":")
        headers[name.strip()] = value.strip()
    if a.api_key_env:
        key = os.environ.get(a.api_key_env)
        if not key:
            sys.exit(f"the environment variable {a.api_key_env} is not set")
        headers["authorization"] = f"Bearer {key}"
    asked, errors = run(items, store, model=a.model, endpoint=a.endpoint, headers=headers, extra=json.loads(a.extra),
                        max_tokens=a.max_tokens, limit=a.limit, redo=a.redo, unread=a.unread, workers=a.workers,
                        save=lambda: out.write_text(dumps(store), encoding="utf-8", newline="\n"))
    print(json.dumps({"model": a.model, "asked": asked, "errors": errors, "families": table(items, store["models"][a.model], a.limit)}, indent=1))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
