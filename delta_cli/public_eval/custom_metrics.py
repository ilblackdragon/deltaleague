"""Metrics for custom JSON tasks; native benchmark tasks keep their native metrics."""
import re


def score_answer(example, text):
    # The shared backend normalizes thinking output before task scoring.
    answer = text.strip()
    scorer = example.get("scorer", "exact")
    if scorer in ("exact", "exact_delimited"):
        separators = example.get("answer_separators") if scorer == "exact_delimited" else None
        if scorer == "exact_delimited" and (
            not isinstance(separators, str) or not separators
            or any(c not in ",;|=/" for c in separators)
        ):
            raise ValueError("exact_delimited requires explicit answer_separators")

        def normalize(s):
            value = re.sub(r"\s+", " ", str(s).strip()).casefold()
            if separators:
                value = re.sub(r"\s*([" + re.escape(separators) + r"])\s*", r"\1", value)
            return value
        return float(normalize(answer) == normalize(example["answer"]))
    if scorer == "choice":
        boxed = re.findall(r"\\boxed\{([A-E1-5])\}", answer)
        matches = boxed or re.findall(r"(?:^|\n)\s*([A-E1-5])[.)]?\s*$", answer)
        return float(bool(matches) and matches[-1] == str(example["answer"]))
    if scorer == "math":
        from math_verify import parse, verify
        from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
        gold = parse(str(example["answer"]), extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        pred = parse(answer, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        return float(bool(gold) and bool(pred) and verify(gold, pred))
    if scorer == "ifeval":
        from lm_eval.tasks.ifeval.utils import process_results
        return float(process_results({**example, 'key': example.get('id', 'custom')}, [answer])['prompt_level_strict_acc'])
    raise ValueError("unknown_scorer")
