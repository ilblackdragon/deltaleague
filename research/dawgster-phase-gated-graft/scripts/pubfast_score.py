"""Score pubfast_gen outputs with the official lm-eval filters/metrics and the WASI code sandbox.

usage: DELTA_WASI_ROOT=pe/wasi .venv-pe/bin/python scripts/pubfast_score.py GEN.jsonl [OUT.json]
Writes per-item scores; prints per-group % plus truncation rate.
"""
import json, sys, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from delta_cli.public_eval.suite import manifest, PRIMARY
from delta_cli.public_eval.generation import final_answer
from delta_cli.public_eval.coding import extract_code, program


def main():
    gen = [json.loads(l) for l in open(sys.argv[1])]
    out = sys.argv[2] if len(sys.argv) > 2 else sys.argv[1].replace('.jsonl', '.scores.json')
    suite = manifest()
    by = {}
    for g in gen:
        by.setdefault(g['group'], {})[g['key']] = g
    res = {}
    from lm_eval.models.openai_completions import LocalCompletionsAPI
    from delta_cli.public_eval import text

    p2k = {}
    for grp in by:
        if grp in suite['text']:
            rep = json.load(open(f'runs/pub_leader/reports/{grp}.json'))
            for t, xs in rep['samples'].items():
                for x in xs:
                    p2k[x['arguments'][0][0]] = f'{t}:{x["doc_id"]}'

    def fake_generate(self, requests, disable_tqdm=False):
        return [final_answer(cur[p2k[r.args[0]]]['text']) for r in requests]
    LocalCompletionsAPI.generate_until = fake_generate
    for grp, cur in by.items():
        if grp in suite['text']:
            globals()['cur'] = cur
            with tempfile.TemporaryDirectory() as td:
                rep = text.run(suite, grp, 'http://127.0.0.1:9/v1', 'candidate', 'models/leader',
                               Path(td) / 'r.json', generation={'enable_thinking': True, 'max_tokens': 8192,
                               'temperature': 1.0, 'top_p': .95, 'top_k': 20, 'presence_penalty': 1.5})
            metric, filt = PRIMARY[grp]
            res[grp] = {f'{t}:{s["doc_id"]}': float(s[metric]) for t in rep['samples'] for s in rep['samples'][t]
                        if s['filter'] == filt}
        else:
            from delta_cli.public_eval.wasi import sandbox
            docs = {str(d['task_id']): d for d in suite['code'][grp]['docs']}

            def one(k):
                code = extract_code(final_answer(cur[k]['text']))
                return k, float(sandbox(program(grp, docs[k], code))['passed'])
            with ThreadPoolExecutor(4) as pool:
                res[grp] = dict(pool.map(one, list(cur)))
    summ = {}
    for grp, sc in res.items():
        cur = by[grp]
        tr = sum('</think>' not in g['text'] for g in cur.values()) / len(cur)
        summ[grp] = {'score': round(100 * sum(sc.values()) / len(sc), 1), 'n': len(sc), 'trunc': round(100 * tr, 1),
                     'mean_tok': round(sum(g['ntok'] for g in cur.values()) / len(cur))}
    allv = [v for sc in res.values() for v in sc.values()]
    summ['_mean_of_groups'] = round(sum(s['score'] for s in summ.values()) / len(res), 2)
    summ['_trunc_all'] = round(100 * sum('</think>' not in g['text'] for g in gen) / len(gen), 1)
    json.dump({'summary': summ, 'items': res}, open(out, 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
