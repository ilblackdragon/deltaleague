"""Run a paired completion check on four hash-verified public math problems."""
import argparse
import hashlib
import json
from pathlib import Path

from delta_cli.local_commands import experiment


def public_problems(workspace):
    from datasets import load_dataset

    raw = (workspace / 'validation.json').read_bytes()
    config = json.loads((workspace / 'evaluation.json').read_text())
    if hashlib.sha256(raw).hexdigest() != config['validation_sha256']:
        raise ValueError('Public validation manifest hash mismatch')
    manifest = json.loads(raw)
    for benchmark, task, repo, subset in (
        ('gsm8k', 'gsm8k', 'openai/gsm8k', 'main'),
        ('math500', 'minerva_math500', 'HuggingFaceH4/MATH-500', 'default'),
    ):
        selection = manifest['text'][benchmark]['selection'][task]
        dataset = load_dataset(repo, subset, split='test',
                               revision=manifest['dataset_revisions'][repo])
        for index, expected in list(zip(selection['indices'], selection['doc_sha256'], strict=True))[:2]:
            document = dataset[index]
            # The Minerva task strips degree units before hashing processed answers.
            if benchmark == 'math500':
                document['answer'] = document['answer'].removesuffix(r'^\circ')
            encoded = json.dumps(document, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
            if hashlib.sha256(encoded.encode()).hexdigest() != expected:
                raise ValueError(f'{benchmark} public document hash mismatch at index {index}')
            question = document['question' if benchmark == 'gsm8k' else 'problem']
            yield {
                'benchmark': benchmark, 'index': index, 'sha256': expected,
                'prompt': question + r' Please reason step by step and put your final answer in \boxed{}.',
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--candidate', action='store_true')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    problems = workspace / 'smoke-problems.json'
    if args.prepare:
        problems.write_text(json.dumps(list(public_problems(workspace)), indent=2))
        return
    cases = json.loads(problems.read_text())
    label = 'candidate' if args.candidate else 'leader'
    weights = workspace / 'edited.safetensors' if args.candidate else None
    with experiment(workspace, None, weights, None, 'mps', None, None) as (runner, details):
        with (workspace / f'{label}-smoke.jsonl').open('x') as output:
            for case in cases:
                result = runner.generate(case['prompt'], details['generation'], seed=1000)
                _, closing, answer = result.pop('text').partition('</think>')
                row = {key: case[key] for key in ('benchmark', 'index', 'sha256')}
                row.update(result, seed=1000, closed_thinking=bool(closing),
                           answer_after_thinking=bool(closing and answer.strip()),
                           answer_tail=answer.strip()[-500:])
                print(json.dumps(row, ensure_ascii=False), file=output, flush=True)
                print(label, row['benchmark'], row['index'], row['tokens'],
                      row['closed_thinking'], row['answer_after_thinking'], flush=True)


if __name__ == '__main__':
    main()
