import json
import os
import statistics
import sys

GENERATION = ("bbh", "gsm8k", "ifeval", "math500", "mmlu_pro")


def main():
    from lm_eval.api.model import TemplateLM
    from lm_eval.evaluator import evaluate
    from vllm import LLM, SamplingParams

    from delta_cli import manifest

    model_dir, workspace, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    config = json.load(open(f"{workspace}/evaluation.json"))
    data = json.load(open(f"{workspace}/validation.json"))
    rows = [r for r in manifest.manifest_examples(data, config["validation_names"])[0] if r["benchmark"] in GENERATION]
    tasks = manifest.load_tasks(rows)
    gen = config["generation"]
    extra = int(os.environ.get("EXTRA", "0"))
    if extra:
        import random
        rng = random.Random(7)
        for name in GENERATION:
            subtasks = sorted({r["subtask"] for r in rows if r["benchmark"] == name})
            for subtask in subtasks:
                taken = {r["doc_index"] for r in rows if r["subtask"] == subtask}
                pool = [i for i in range(len(tasks[subtask].eval_docs)) if i not in taken]
                for i in rng.sample(pool, min(len(pool), -(-extra // len(subtasks)))):
                    rows.append({"benchmark": name, "subtask": subtask, "doc_index": i, "example_id": f"extra-{subtask}-{i}"})

    llm = LLM(model_dir, dtype="bfloat16", max_model_len=gen["max_model_len"], seed=0, gpu_memory_utilization=float(os.environ.get("GPU_MEM", "0.7")), max_num_seqs=int(os.environ.get("MAX_SEQS", "256")))
    tok = llm.get_tokenizer()
    params = SamplingParams(temperature=gen["temperature"], top_p=gen["top_p"], top_k=gen["top_k"],
                            presence_penalty=gen["presence_penalty"], max_tokens=gen["max_tokens"], seed=1000,
                            skip_special_tokens=False)
    samples, generated = {}, {}
    for row in rows:
        samples.setdefault(row["subtask"], []).append(row["doc_index"])

    class VLLMLM(TemplateLM):
        @property
        def eot_token_id(self):
            return tok.eos_token_id

        def tok_encode(self, string, **kwargs):
            return tok.encode(string, add_special_tokens=False)

        def _loglikelihood_tokens(self, requests, disable_tqdm=False):
            raise NotImplementedError

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            raise NotImplementedError

        def generate_until(self, requests, disable_tqdm=False):
            prompts = [tok.apply_chat_template([{"role": "user", "content": r.args[0]}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=gen["enable_thinking"])
                       for r in requests]
            outputs = llm.generate(prompts, params)
            texts = []
            for r, o in zip(requests, outputs):
                generated[r.task_name, samples[r.task_name][r.doc_id]] = o.outputs[0]
                texts.append(o.outputs[0].text.split("</think>")[-1].strip())
            return texts

    output = evaluate(VLLMLM(), {name: tasks[name] for name in samples}, samples=samples, bootstrap_iters=0,
                      log_samples=True, verbosity="WARNING")
    scored = {}
    for subtask, entries in output["samples"].items():
        for entry in entries:
            values = {}
            for m in [x["metric"] for x in tasks[subtask].config.metric_list or []]:
                if m in entry:
                    v = entry[m]
                    values[m] = float(statistics.mean(v)) if isinstance(v, list) else float(v)
            scored.setdefault((subtask, entry["doc_id"]), {})[entry["filter"]] = values
    per, closed = {}, {}
    with open(out_path, "w") as f:
        for row in rows:
            key = row["subtask"], row["doc_index"]
            out = generated[key]
            for flt, values in scored[key].items():
                for metric, value in values.items():
                    per.setdefault((row["benchmark"], flt, metric), []).append(value)
            closed.setdefault(row["benchmark"], []).append("</think>" in out.text)
            f.write(json.dumps({"benchmark": row["benchmark"], "example_id": row["example_id"], "scores": scored[key],
                                "tokens": len(out.token_ids), "closed": "</think>" in out.text, "text": out.text}) + "\n")
    for (name, flt, metric), values in per.items():
        print(f"{name:10} {flt:18} {metric:24} n={len(values)} score={100 * statistics.mean(values):.2f} "
              f"closed={statistics.mean(closed[name]):.3f}")
    print("generation mean=done")


if __name__ == "__main__":
    main()
