"""Evidence verdicts (benchmarks/public/evidence-verdicts-v1): the committed set is what its builder makes,
it runs through `delta evaluate`, and each safeguard its README states holds for every item."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from delta_cli import local, main, scoring
from delta_cli.public_eval import custom_metrics
from delta_cli.public_eval.generation import final_answer

HERE = Path(__file__).parents[1] / 'benchmarks/public/evidence-verdicts-v1'
SET = HERE / 'evidence-verdicts-v1.json'
LETTERS = 'ABC'


def module(name):
    spec = importlib.util.spec_from_file_location(f'evidence_verdicts_{name}', HERE / f'{name}.py')
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


build_set, breakdown = module('build_set'), module('breakdown')
CFG, ITEMS, PROV = read('config.json'), read('evidence-verdicts-v1.json'), read('provenance.json')
RECORDS = {r['id']: r for r in read('records.json')['records']}
ROWS = [(family, item, PROV[item['id']]) for family, rows in ITEMS.items() for item in rows]


def shown(prompt):
    """The claim and the (host, passage) pairs a prompt shows."""
    lines = prompt.split('\n')
    passages = []
    for line in lines[4:]:
        found = re.fullmatch(r'\[\d+\] (\S+): "(.*)"', line)
        if not found:
            break
        passages.append(found.groups())
    return lines[1], passages


def values(texts):
    return set().union(*(build_set.value_set(t) for t in texts))


def test_committed_files_are_what_the_builder_makes():
    files = build_set.build(read('records.json')['records'], CFG, read('solvers.json'), read('handcheck.json'))
    assert sorted(files) == ['dropped.json', 'evidence-verdicts-v1.json', 'evidence-verdicts-v1.sha256',
                             'provenance.json', 'report.json', 'vetoed.json']
    for name, text in files.items():
        assert (HERE / name).read_bytes() == text.encode('utf-8'), name
    digest = hashlib.sha256(SET.read_bytes()).hexdigest()
    assert (HERE / 'evidence-verdicts-v1.sha256').read_text() == f'{digest}  evidence-verdicts-v1.json\n'
    assert read('report.json')['sha256'] == digest


def test_loads_with_the_platform_loader_and_only_the_gold_letter_scores(tmp_path):
    tasks, names, full, sha = local.public_examples(tmp_path, None, validation_file=SET)
    assert names == build_set.FAMILIES and not full and sha == hashlib.sha256(SET.read_bytes()).hexdigest()
    assert len(tasks) == len(ROWS) == len({item['id'] for _, item, _ in ROWS})
    for task in tasks:
        example = task['example']
        assert list(example) == ['id', 'prompt', 'answer', 'scorer', 'samples'] and example['scorer'] == 'choice'
        assert example['prompt'].endswith('\n' + CFG['instruction'])
        for letter in LETTERS:
            text = f'The passage settles it.\n\\boxed{{{letter}}}'
            want = float(letter == example['answer'])
            assert custom_metrics.score_answer(example, text) == want
            # The platform scorer drops a thinking trace first, whatever letter the trace held.
            assert scoring.score_answer(example, f'<think>\\boxed{{{example["answer"]}}}</think>{text}') == want


def test_no_constant_answer_scores():
    # One label per family: always giving one label gets one family right and two wrong.
    assert all(p['label'] == build_set.LABEL[family] for family, _, p in ROWS)
    # One letter: an item's option order comes from a hash of its key, so each letter is gold about a third of the time.
    for letter in LETTERS:
        shares = [sum(item['answer'] == letter for item in rows) / len(rows) for rows in ITEMS.values()]
        assert all(.2 < share < .47 for share in shares) and abs(sum(shares) / 3 - 1 / 3) < .03, (letter, shares)


def test_every_item_is_what_its_provenance_says():
    people = re.compile(CFG['people'])
    for family, item, p in ROWS:
        assert item['id'] == p['key'].replace('|', '.') and 'ev_' + p['family'] == family
        assert sorted(p['options']) == sorted(CFG['labels']) and p['options'][LETTERS.index(item['answer'])] == p['label']
        assert p['prompt_sha256'] == hashlib.sha256(item['prompt'].encode()).hexdigest()
        wikipedia = p['wikipedia']
        assert re.fullmatch(r'https://en\.wikipedia\.org/w/index\.php\?title=[^&]+&oldid=\d+', wikipedia['url'])
        assert wikipedia['licence'] == 'CC BY-SA 4.0' and wikipedia['sentence_id'] in RECORDS
        claim, passages = shown(item['prompt'])
        assert 0 < len(claim) <= CFG['caps']['claim_chars'] and not people.search(claim)
        # The prompt shows what the saved record holds: each passage is the receipt's excerpt, named by host only.
        lender = RECORDS[p['evidence_from']]
        lent = {(r['key'], r['pattern']): r['excerpt'] for r in lender['receipts'] + [lender['near'] or {}] if r}  # one source can give two receipts
        assert passages == [(e['host'], lent[e['key'], e['pattern']]) for e in p['evidence']]
        assert 1 <= len(passages) <= CFG['caps']['excerpts_per_item']
        for e in p['evidence']:
            assert e['url'].startswith('https://') and re.fullmatch(r'[0-9a-f]{64}', e['sha256']) and e['fetched_at']
            assert e['kind'] in CFG['kinds_in_prompts'] and len(lent[e['key'], e['pattern']]) <= CFG['caps']['excerpt_chars']
            assert not any(e['host'] == host or e['host'].endswith('.' + host) for host in CFG['excluded_hosts'])


def test_a_supported_item_is_a_checked_claim_with_its_own_passages():
    for family, item, p in ROWS:
        if family != 'ev_supported':
            continue
        record = RECORDS[p['wikipedia']['sentence_id']]
        assert record['verdict'] == 'current' and not p['claim']['modified']
        assert p['evidence_from'] == record['id'] == p['cluster'] and shown(item['prompt'])[0] == build_set.claim_text(record, CFG)


def test_a_refuted_item_is_a_caught_line_or_one_safe_edit():
    edited = 0
    for family, item, p in ROWS:
        if family != 'ev_refuted':
            continue
        record = RECORDS[p['wikipedia']['sentence_id']]
        claim, passages = shown(item['prompt'])
        assert p['evidence_from'] == record['id'] == p['cluster']
        if not p['claim']['modified']:
            # A line the corpus caught: stale or wrong as written, shown with the passage that says so.
            assert record['verdict'] in ('stale', 'wrong') and p['stratum'] == record['verdict'] and claim == build_set.claim_text(record, CFG)
            continue
        edited += 1
        edit, original = p['claim']['edit'], p['claim']['original']
        assert record['verdict'] == 'current' and original == build_set.claim_text(record, CFG)
        was, now = f"{edit['from_value']}|{edit['unit']}", f"{edit['to_value']}|{edit['unit']}"
        in_passages = values(text for _, text in passages)
        # The passages state the true value and never the edited one; the claim changed in that one value only.
        assert was in in_passages and now not in in_passages
        assert build_set.value_set(claim) == (build_set.value_set(original) - {was}) | {now} != build_set.value_set(original)
        if edit['rule'] == 'leading-digit':
            assert p['stratum'] == 'numeric_edit'
            a, b = build_set.common(edit['to_value'], edit['from_value'])
            assert 4 * a <= 3 * b or 4 * a >= 5 * b  # at least a quarter away from the true value
            if edit['unit'] == '%':
                share, whole = build_set.common(edit['to_value'], '100')
                assert share <= whole
        else:
            assert edit['rule'] == 'year-shift' and p['stratum'] == 'year_edit' and record['class'] == 'sourced'
            assert 2 <= abs(int(edit['to_value']) - int(edit['from_value'])) <= 7
            assert CFG['years']['min'] <= int(edit['to_value']) <= CFG['years']['max']
    assert edited > 0


def test_an_edit_skips_ranges_and_dated_bounds():
    # one end of a range, with a word or a scale word between the end and "to"
    for claim in ('The service had 3 million users to 9 million users that year.', 'Fees rose from 5 percent to 10 percent in 2023.'):
        edits, skipped = build_set.numeric_edits('x-1', claim, [claim])
        assert edits == [] and {row['why'] for row in skipped} == {'the value is one end of a range'}, claim
    # a year after a bound word, with a month, a day or a quarter between them
    passages = ['It was audited in 2024; the rule held in 2020.']
    for claim in ('The fund was audited through Q1 2024 by the firm.', 'The rule held until June 2020, the regulator said.'):
        edits, skipped = build_set.year_edits('x-1', claim, passages)
        assert edits == [] and [row['why'] for row in skipped] == ['the year follows a word that makes it a bound or a span'], claim
    # and the same year with no bound before it is edited
    assert build_set.year_edits('x-1', 'The fund was audited in 2024 by the firm.', passages)[0]


def test_an_insufficient_item_shows_passages_that_share_nothing_with_its_claim():
    lenders = []
    for family, item, p in ROWS:
        if family != 'ev_insufficient' or p['stratum'] in build_set.HARD:
            continue
        record, lender = RECORDS[p['wikipedia']['sentence_id']], RECORDS[p['evidence_from']]
        claim, passages = shown(item['prompt'])
        lenders.append(lender['id'])
        assert record['verdict'] == 'current' and lender['verdict'] == 'current' and lender['id'] != record['id']
        assert claim == build_set.claim_text(record, CFG) and not p['claim']['modified'] and p['cluster'] == record['id']
        assert not {r['url'] for r in record['receipts']} & {e['url'] for e in p['evidence']}
        assert not build_set.value_set(claim) & values(text for _, text in passages)
        same_page = record['article']['slug'] == lender['article']['slug']
        assert p['stratum'] == ('same_page' if same_page else 'cross_page')
        assert len(p.get('shared_words', [])) <= (1 if same_page else 0)
    assert len(lenders) == len(set(lenders)) <= CFG['caps']['insufficient_items']  # a record lends its passages once


def test_a_close_insufficient_item_shares_words_and_leaves_the_value_out():
    people, lenders, seen, decoys = re.compile(CFG['people']), [], {'near_miss': 0, 'related': 0}, 0
    for family, item, p in ROWS:
        if p['stratum'] not in build_set.HARD:
            continue
        assert family == 'ev_insufficient'
        seen[p['stratum']] += 1
        record, lender = RECORDS[p['wikipedia']['sentence_id']], RECORDS[p['evidence_from']]
        claim, passages = shown(item['prompt'])
        assert record['verdict'] == 'current' and claim == build_set.claim_text(record, CFG) and p['cluster'] == record['id']
        assert len(passages) == 1 and not people.search(passages[0][1])
        text, title_words = passages[0][1], build_set.title_words_of(record['article']['title'])
        # The claim states a quantity or a year, and the passage holds none of its values, in any way of writing them.
        assert build_set.value_set(claim) and not build_set.value_set(claim) & build_set.value_set(text)
        assert not any(digits in re.sub(r'[^0-9]+', '', text) for digits in build_set.bare_digits(claim))
        # It is close: two or more content words in common, beyond the article's own title and the scale and approximation words.
        in_claim = set(build_set.content_words(claim, title_words)) - set(CFG['close_ignore_words'])
        assert p['shared_words'] == sorted(w for w in build_set.content_words(text, title_words) if w in in_claim) and len(p['shared_words']) >= 2
        # A decoy is another value of the same kind as one of the claim's.
        assert p['decoy'] == bool(build_set.value_kinds(text) & build_set.value_kinds(claim))
        decoys += p['decoy']
        own_pages = {r['url'] for r in record['receipts']}
        if p['stratum'] == 'near_miss':
            # another sentence of the claim's own source, outside every passage the claim's own item shows
            assert lender['id'] == record['id'] and p['evidence'][0]['url'] in own_pages and p['evidence'][0]['cut'] == 'near'
            assert not any(r['excerpt'] and (text in r['excerpt'] or r['excerpt'] in text) for r in record['receipts'])
        else:
            # a passage of another record of the same article, from another page
            lenders.append(lender['id'])
            assert lender['id'] != record['id'] and lender['verdict'] == 'current' and lender['article']['slug'] == record['article']['slug']
            assert p['evidence'][0]['url'] not in own_pages
    assert min(seen.values()) > 0 and sum(seen.values()) <= CFG['caps']['hard_insufficient_items'] and len(lenders) == len(set(lenders))
    assert 0 < decoys < sum(seen.values())


def tampered(record_id, **near):
    """The records with one record's close passage changed, as a hand edit would change it."""
    records = json.loads(json.dumps(read('records.json')['records']))
    record = next(r for r in records if r['id'] == record_id)
    record['near'] = {**record['near'], **near}
    return records


def test_the_builder_vetoes_a_close_passage_that_breaks_the_rule():
    target = next(p for p in PROV.values() if p['stratum'] == 'near_miss')['cluster']
    record = RECORDS[target]
    key = f'ins|{target}|near'
    supporting = next(r['excerpt'] for r in record['receipts'] if r['excerpt'])
    cases = {
        # the claim's own supporting passage: it states the claimed value, and it is what the supported item shows
        supporting: "its near-miss passage holds one of the claim's values",
        'The weather in the region was mild and pleasant for most of that whole season.': 'its near-miss passage shares fewer than two content words with the claim',
    }
    for excerpt, why in cases.items():
        items, vetoed, _ = build_set.candidates(tampered(target, excerpt=excerpt), CFG)
        assert key not in {x['key'] for x in items}
        assert {'key': key, 'family': 'ev_insufficient', 'cluster': target, 'why': why} in vetoed


def test_near_miss_picks_a_close_sentence_and_prefers_a_decoy(tmp_path):
    near_miss = module('near_miss')
    claim, title = 'the network upgrade went live on mainnet in March 2024', 'Example chain'
    shown = ['The upgrade went live on mainnet in March 2024, as planned by the core team.']
    page = ' '.join([
        'The upgrade went live on mainnet in March 2024, as planned by the core team.',      # shown already, and holds the value
        'Every network upgrade is tested on public testnets for several weeks first.',       # close, no decoy
        'An earlier network upgrade had gone live on mainnet in June 2022 after delays.',    # close, with a decoy year
        'The foundation published its annual report in the following spring season.',        # not close
    ])
    hit = near_miss.near_passage(page, claim, title, shown)
    assert hit['excerpt'] == 'An earlier network upgrade had gone live on mainnet in June 2022 after delays.' and hit['decoy']
    assert re.search(hit['pattern'], page) and hit['shared'] == sorted(set(hit['shared'])) and len(hit['shared']) >= 2
    # without the decoy sentence, the close sentence with no decoy is chosen
    second = near_miss.near_passage(page.replace('An earlier network upgrade had gone live on mainnet in June 2022 after delays. ', ''), claim, title, shown)
    assert second['excerpt'].startswith('Every network upgrade') and not second['decoy']
    # a claim with no value gets no close passage, and tidy() decodes what a saved page kept
    assert near_miss.near_passage(page, 'the upgrade went live on mainnet', title, shown) is None
    assert near_miss.tidy('Tether&#8217;s  reserves &amp; moreÂ here') == 'Tether’s reserves & more here'
    # the whole record: across receipts, the same order
    (tmp_path / 'k1.txt').write_text(page, encoding='utf-8')
    record = {'verdict': 'current', 'claim': claim, 'article': {'title': title}, 'receipts': [
        {'key': 'k1', 'url': 'https://a.example/1', 'host': 'a.example', 'kind': 'prose', 'fetched_at': 'then', 'status': 200, 'sha256': '0' * 64, 'excerpt': shown[0]}]}
    near = near_miss.near_for_record(record, lambda key: (tmp_path / f'{key}.txt').read_text(encoding='utf-8'), CFG)
    assert near['excerpt'] == hit['excerpt'] and near['key'] == 'k1' and set(near) == {'key', 'url', 'host', 'kind', 'fetched_at', 'status', 'sha256', 'pattern', 'excerpt'}


def test_a_close_passage_names_no_listed_person_and_scale_words_are_not_shared():
    near_miss = module('near_miss')
    claim, title = 'the exchange handled about 5 million orders a day', 'Example chain'
    page = ('The exchange said that nearly a million people had waited about an hour. '
            'Vitalik said the exchange handled orders for 8 million people in 2023.')
    people, ignore = re.compile(CFG['people'], re.A), set(CFG['close_ignore_words'])
    assert {'million', 'about'} <= ignore and people.search('Vitalik said')
    # with neither rule, the sentence with a decoy wins
    assert near_miss.near_passage(page, claim, title, [])['excerpt'].startswith('Vitalik said')
    # a sentence that names someone on the list is skipped
    hit = near_miss.near_passage(page, claim, title, [], people=people)
    assert hit['excerpt'].startswith('The exchange said') and hit['shared'] == ['exchange', 'million']
    # and "million" is not a shared word, so what is left shares one word with the claim: no close passage
    assert near_miss.near_passage(page, claim, title, [], ignore=ignore, people=people) is None


def gate(letters, handcheck=None, rule='sibling_not_refuted'):
    """final_set over one true claim and one edit of it, with the given letters from two gate models."""
    cfg = {'gate_models': ['one', 'two'], 'edit_rule': rule, 'caps': {'insufficient_items': 100}}
    order = ['supported', 'refuted', 'insufficient']  # so A is "supported" and B is "refuted" on both items
    plain = {'key': 'sup|x-1', 'family': 'ev_supported', 'sha': 'p', 'answer': 'A', 'options': order}
    edit = {'key': 'ref|x-1|leading-digit|0', 'family': 'ev_refuted', 'sha': 'e', 'answer': 'B', 'options': order, 'needs': 'sup|x-1'}
    solvers = {'models': {model: {sha: {'letter': letter} for sha, letter in asked.items() if letter != 'unasked'}
                          for model, asked in zip(cfg['gate_models'], letters)}}
    kept, dropped = build_set.final_set([plain, edit], cfg, solvers, handcheck or {'rows': []})
    return [item['key'] for item in kept], {row['key']: row['why'] for row in dropped}


def test_the_gate_needs_every_gate_model_and_the_hand_check_can_only_remove():
    both = ['sup|x-1', 'ref|x-1|leading-digit|0']
    assert gate([{'p': 'A', 'e': 'B'}, {'p': 'A', 'e': 'B'}]) == (both, {})
    # Either model alone is enough to drop an item: a wrong letter, no readable letter, or never asked.
    for other in ('C', None, 'unasked'):
        for letters in ([{'p': 'A', 'e': other}, {'p': 'A', 'e': 'B'}], [{'p': 'A', 'e': 'B'}, {'p': 'A', 'e': other}]):
            kept, dropped = gate(letters)
            assert kept == ['sup|x-1'] and list(dropped) == ['ref|x-1|leading-digit|0']
    assert 'no letter the scorer can read' in gate([{'p': 'A', 'e': None}, {'p': 'A', 'e': 'B'}])[1][both[1]]
    assert 'has not been asked' in gate([{'p': 'A', 'e': 'unasked'}, {'p': 'A', 'e': 'B'}])[1][both[1]]
    # An edit also needs its unedited claim: under the rule in use, no gate model may call that claim refuted.
    kept, dropped = gate([{'p': 'A', 'e': 'B'}, {'p': 'C', 'e': 'B'}])
    assert kept == [both[1]] and list(dropped) == ['sup|x-1']  # "not enough evidence" on the unedited claim is not "refuted"
    assert gate([{'p': 'A', 'e': 'B'}, {'p': 'B', 'e': 'B'}])[0] == []
    assert gate([{'p': 'A', 'e': 'B'}, {'p': 'C', 'e': 'B'}], rule='sibling_supported')[0] == []
    # The hand check never relabels. A disagreement or a flag removes the item, and an edit goes with its claim.
    for row in ({'key': 'sup|x-1', 'prompt_sha256': 'p', 'agrees': False}, {'key': 'sup|x-1', 'prompt_sha256': 'p', 'agrees': True, 'flag': 'two readings'}):
        assert gate([{'p': 'A', 'e': 'B'}, {'p': 'A', 'e': 'B'}], {'rows': [row]})[0] == []
    assert gate([{'p': 'A', 'e': 'B'}, {'p': 'A', 'e': 'B'}], {'rows': [{'key': both[1], 'prompt_sha256': 'e', 'agrees': False}]})[0] == ['sup|x-1']
    assert gate([{'p': 'A', 'e': 'B'}, {'p': 'A', 'e': 'B'}], {'rows': [{'key': 'sup|x-1', 'prompt_sha256': 'p', 'agrees': True}]})[0] == both
    # A row counts only for the prompt it was read on: with no hash, or another prompt's, it removes nothing.
    for row in ({'key': 'sup|x-1', 'agrees': False}, {'key': 'sup|x-1', 'prompt_sha256': 'e', 'agrees': False}):
        assert gate([{'p': 'A', 'e': 'B'}, {'p': 'A', 'e': 'B'}], {'rows': [row]})[0] == both


def test_the_breakdown_reads_an_answer_as_the_evaluator_scores_it():
    example = {'answer': 'B', 'scorer': 'choice'}
    answers = ['\\boxed{B}', 'So it is refuted.\nB', 'First \\boxed{A}, on reflection \\boxed{B}', 'The answer is B.', '',
               '<think>\\boxed{B}', '<think>\\boxed{A}</think>\\boxed{B}', '<think>a</think>\\boxed{B}</think>\\boxed{C}', 'B)\n']
    for text in answers:
        for thinking in (False, True):
            scored = custom_metrics.score_answer(example, final_answer(text) if thinking else text)
            assert (breakdown.read_choice(breakdown.final_text(text, thinking)) == 'B') == bool(scored), (text, thinking)


class Abstainer:
    """Stands in for a model: always picks "not enough evidence", whatever letter that option has."""
    details = {'device': 'cpu'}
    reply = 'I cannot tell from this.\n\\boxed{%s}'

    def __init__(self, *args):
        pass

    def generate(self, prompt, generation, seed):
        letter = next(line[0] for line in prompt.split('\n') if line[1:].startswith('. Not enough evidence'))
        return {'text': self.reply % letter, 'tokens': 9, 'generation_seconds': .1, 'truncated': False}


class Unfinished(Abstainer):
    reply = '<think>The passage gives another figure, so \\boxed{%s}'


@pytest.mark.parametrize('runner, thinking, right', [(Abstainer, '--no-thinking', 3), (Unfinished, '--thinking', 0)])
def test_runs_through_delta_evaluate_and_the_breakdown_agrees(workspace, monkeypatch, runner, thinking, right):
    root, _, _ = workspace
    monkeypatch.setattr(local, 'Runner', runner)
    output = root / 'run.json'
    result = CliRunner().invoke(main.app, ['evaluate', '--workspace', str(root), '--validation', str(SET), '--limit', '3',
                                           '--max-new-tokens', '64', thinking, '--output', str(output)])
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    assert report['validation_sha256'] == hashlib.sha256(SET.read_bytes()).hexdigest() and report['validation_kind'] == 'custom'
    assert report['validation_benchmarks'] == {'ev_supported': 0, 'ev_refuted': 0, 'ev_insufficient': right / 3 * 100}
    saved = breakdown.answers_of_report(output, SET)
    kept = {'benchmark', 'prompt_sha256', 'sample', 'letter', 'score', 'tokens', 'truncated'}  # no generated text, no paths
    assert 'validation_file' not in saved['run'] and all(set(row) == kept for row in saved['answers'])
    records, absent, differs = breakdown.tally(saved['answers'], PROV)
    summary = breakdown.summarise(records)
    assert (len(records), absent, differs) == (9, 0, 0)
    assert {name: family['right'] for name, family in summary['families'].items()} == {
        'ev_supported': 0, 'ev_refuted': 0, 'ev_insufficient': right}
    assert summary['rules_without_evidence']['count'] == 0
    assert summary['declines_a_clear_case']['count'] == (6 if right else 0)
    assert summary['no_answer']['count'] == (0 if right else 9)  # thinking that never closes is never scored


def test_saved_baselines_keep_their_reports_numbers():
    names = sorted(p.name for p in (HERE / 'baselines').glob('*.answers.json'))
    assert names, 'no baseline run is saved'
    for name in names:
        saved = read(f'baselines/{name}')
        run, answers = saved['run'], saved['answers']
        assert 'validation_file' not in run and 'generations_file' not in run  # no machine paths
        records, absent, differs = breakdown.tally(answers, PROV)
        assert differs == 0 and len(records) + absent == len(answers)
        # The report's own family scores follow from the saved per-item scores.
        for family in run['selected_benchmarks']:
            scores = [row['score'] for row in answers if row['benchmark'] == family]
            assert sum(scores) / len(scores) * 100 == pytest.approx(run['validation_benchmarks'][family]), (name, family)


def test_the_human_read_was_of_a_mixed_sample_fixed_by_hash():
    hand_check = module('hand_check')
    pool = hand_check.pool_of(HERE)
    flagged = hand_check.flagged_keys(HERE, pool)
    plan = read('handcheck-sample.json')
    picked = hand_check.sample(pool, plan['per_part'], flagged)
    # The page showed exactly this sample, in this order: every part by hash, and every flagged item, mixed.
    assert [x['key'] for x in picked] == [x['key'] for x in plan['items']]
    assert [hand_check.part_of(x, flagged) for x in picked] == [x['part'] for x in plan['items']]
    assert [x['sha'] for x in picked] == [x['prompt_sha256'] for x in plan['items']]
    parts = [hand_check.part_of(x, flagged) for x in picked]
    assert all(parts.count(name) == plan['per_part'] for name, _ in hand_check.PARTS) and parts.count('flagged') == len(flagged)
    assert {x['key'] for x in picked} <= {x['key'] for x in hand_check.sample(pool, plan['per_part'] + 2, flagged)}  # a longer read continues the same order
    # The flagged items are the ones flagged.json names and the ones a model outside the gate answered differently.
    assert {r['key'] for r in read('flagged.json')['rows']} <= flagged
    store = read('handcheck.json')
    rows = store['rows']
    by_key = {x['key']: x for x in pool}
    for row in rows:
        item = by_key[row['key']]
        assert row['letter'] in ('A', 'B', 'C', None) and (row['flag'] is not None) == (row['letter'] is None)
        assert row['prompt_sha256'] == item['sha'] and row['part'] == hand_check.part_of(item, flagged)
        assert row['agrees'] == (row['letter'] == item['answer'] if row['letter'] else None)
        assert row['label'] == (item['options'][LETTERS.index(row['letter'])] if row['letter'] else None)
    assert not rows or [row['key'] for row in rows] == [x['key'] for x in plan['items']]
    # Every item a reader disagreed with or could not decide is out of the set, and so is an edit of a removed claim.
    removed = {row['key'] for row in rows if row['agrees'] is False or row['flag']}
    kept = {p['key'] for _, _, p in ROWS}
    assert not removed & kept
    assert read('report.json')['hand_check'] == {'rows': len(rows), 'agreed': sum(row['agrees'] is True for row in rows),
                                                 'disagreed': sum(row['agrees'] is False for row in rows),
                                                 'flagged': sum(bool(row['flag']) for row in rows), 'rows_for_a_changed_or_gone_item': 0}
    # A set-aside sitting is kept for the record and changes nothing.
    for aside in store.get('set_aside', []):
        assert aside['why'] and aside['rows']
        assert build_set.build(read('records.json')['records'], CFG, read('solvers.json'), {'rows': rows}) == \
            build_set.build(read('records.json')['records'], CFG, read('solvers.json'), store)


def test_a_row_counts_only_for_the_prompt_it_was_read_on():
    item = next(x for x in build_set.candidates(read('records.json')['records'], CFG)[0] if x['family'] == 'ev_supported')
    now = {item['key']: item}
    assert build_set.row_applies({'key': item['key'], 'prompt_sha256': item['sha']}, now)
    assert not build_set.row_applies({'key': item['key'], 'prompt_sha256': '0' * 64}, now)
    assert not build_set.row_applies({'key': 'sup|no-such-record', 'prompt_sha256': item['sha']}, now)
    assert not build_set.row_applies({'key': item['key']}, now)  # a row that does not say which prompt it was read on counts for nothing
    assert not build_set.row_applies({'key': item['key'], 'prompt_sha256': None}, now)


def test_hand_check_records_what_the_reader_picks(tmp_path):
    hand_check = module('hand_check')
    for name in ('config.json', 'records.json', 'solvers.json', 'flagged.json'):
        (tmp_path / name).write_bytes((HERE / name).read_bytes())
    pool = hand_check.pool_of(tmp_path)
    flagged = hand_check.flagged_keys(tmp_path, pool)
    picked = hand_check.sample(pool, 1, flagged)
    answers = iter([picked[0]['answer'], 'x', '?', 'two readings', 'q'])
    shown = []
    assert hand_check.main(['--dir', str(tmp_path), '--reader', 'r9', '--per', '1'], ask=lambda _: next(answers), say=shown.append) == 0
    rows = json.loads((tmp_path / 'handcheck.json').read_text())['rows']
    assert [(row['key'], row['agrees'], row['flag']) for row in rows] == [(picked[0]['key'], True, None), (picked[1]['key'], None, 'two readings')]
    assert [row['prompt_sha256'] for row in rows] == [picked[0]['sha'], picked[1]['sha']]
    assert [row['part'] for row in rows] == [hand_check.part_of(x, flagged) for x in picked[:2]]
    text = '\n'.join(shown)
    assert hand_check.RULE in text and picked[0]['prompt'].split('\n')[1] in text and CFG['instruction'] not in text
    assert not any(label in text for label in ('ev_supported', 'ev_refuted', 'ev_insufficient', 'flagged'))  # neither family nor reason is shown




def test_ask_models_asks_each_prompt_once_and_keeps_only_the_letter(tmp_path, monkeypatch):
    import http.server
    import threading
    ask_models = module('ask_models')
    seen, failing = [], [0]

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['content-length'])))
            seen.append((self.path, self.headers, body))
            if failing[0]:
                failing[0] -= 1
                self.send_response(503)
                self.end_headers()
                return
            text = 'It goes on' if 'second' in body['messages'][0]['content'] else 'Reasoning.\n\\boxed{B}'
            reply = json.dumps({'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}).encode()
            self.send_response(200)
            self.send_header('content-length', str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        endpoint = f'http://127.0.0.1:{server.server_port}/v1'
        items = {'family': [{'id': 'a', 'prompt': 'first', 'answer': 'B'}, {'id': 'b', 'prompt': 'second', 'answer': 'A'}]}
        (tmp_path / 'items.json').write_text(json.dumps(items))
        out = tmp_path / 'solvers.json'
        monkeypatch.setenv('EV_TEST_KEY', 'not-a-real-key')
        args = [str(tmp_path / 'items.json'), '--endpoint', endpoint, '--model', 'm', '--api-key-env', 'EV_TEST_KEY', '--workers', '1',
                '--extra', '{"chat_template_kwargs": {"enable_thinking": false}, "max_tokens": 7}', '--header', 'x-no-aliasing: true',
                '--out', str(out)]
        assert ask_models.main(args) == 0
        path, headers, body = seen[0]
        assert path == '/v1/chat/completions' and headers['authorization'] == 'Bearer not-a-real-key' and headers['x-no-aliasing'] == 'true'
        # One user message at temperature 0; the extras are merged, and the tool's own token cap wins.
        assert body == {'model': 'm', 'messages': [{'role': 'user', 'content': 'first'}], 'temperature': 0,
                        'chat_template_kwargs': {'enable_thinking': False}, 'max_tokens': 4096, 'stream': False}
        key = lambda prompt: hashlib.sha256(prompt.encode()).hexdigest()
        assert json.loads(out.read_text())['models']['m'] == {key('first'): {'letter': 'B', 'finish': 'stop', 'chars': 20},
                                                              key('second'): {'letter': None, 'finish': 'stop', 'chars': 10}}
        assert 'not-a-real-key' not in out.read_text()
        # A second run asks nothing; --unread asks again only where no letter was read.
        assert ask_models.main(args) == 0 and len(seen) == 2
        assert ask_models.main(args + ['--unread']) == 0 and len(seen) == 3
        # A server error is retried. A request that keeps failing is not an answer: nothing is stored for it.
        failing[0] = 1
        assert ask_models.ask(endpoint + '/chat/completions', {}, {'messages': [{'content': 'first'}]}, wait=0)['letter'] == 'B'
        failing[0] = 99
        store = {}
        assert ask_models.run(items, store, model='m', endpoint=endpoint, headers={}, extra={}, max_tokens=5, workers=1, wait=0) == (2, 2)
        assert store == {'models': {'m': {}}}
    finally:
        server.shutdown()
        server.server_close()


def test_ask_models_sends_the_key_nowhere_else_and_stores_only_real_answers():
    import http.server
    import threading
    ask_models = module('ask_models')
    elsewhere, replies = [], []

    class Elsewhere(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            elsewhere.append(dict(self.headers))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass
    other = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Elsewhere)

    class Api(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['content-length']))
            kind = replies.pop(0)
            if kind == 'redirect':
                self.send_response(302)
                self.send_header('location', f'http://127.0.0.1:{other.server_port}/steal')
                self.end_headers()
                return
            body = {'error': {'message': 'overloaded'}} if kind == 'error body' else [1, 2] if kind == 'a list' else {'choices': [{'message': {}}]}
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header('content-length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass
    api = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Api)
    for s in (other, api):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        url = f'http://127.0.0.1:{api.server_port}/v1/chat/completions'
        # A redirect is not followed, so the key never reaches the other server.
        replies.append('redirect')
        assert 'redirect' in ask_models.ask(url, {'authorization': 'Bearer not-a-real-key'}, {'messages': []}, wait=0)['error']
        assert elsewhere == []
        # A 200 that is not a chat completion with text is not an answer.
        for kind in ('error body', 'a list', 'no text'):
            replies.append(kind)
            assert 'error' in ask_models.ask(url, {}, {'messages': []}, tries=1, wait=0), kind
    finally:
        for s in (other, api):
            s.shutdown()
            s.server_close()
    # A model that spent its whole budget thinking returns no text: an answer with no letter, kept with its finish reason.
    assert ask_models.reading({'choices': [{'message': {'content': None}, 'finish_reason': 'length'}]}) == {'letter': None, 'finish': 'length', 'chars': 0}
    assert 'error' in ask_models.reading({'choices': [{'message': {'content': None}}]})
    # https only, except for a server on this machine
    assert ask_models.check_endpoint('https://cloud-api.near.ai/v1') and ask_models.check_endpoint('http://127.0.0.1:8000/v1')
    with pytest.raises(ValueError):
        ask_models.check_endpoint('http://cloud-api.near.ai/v1')


def test_check_receipts_tests_the_receipt_and_the_shown_passage():
    check_receipts = module('check_receipts')
    receipts = check_receipts.receipts_of(HERE)
    assert len(receipts) == len({(e['url'], e['pattern']) for _, _, p in ROWS for e in p['evidence']}) > 0
    assert all(row['passage'] for row in receipts.values())
    page = b'<html><style>p{}</style><p>Launched in&nbsp;<b>April   2014</b>, it grew.</p><script>var x = "June 2019"</script></html>'
    assert check_receipts.to_text(page.decode()) == 'Launched in April 2014 , it grew.'
    saved = hashlib.sha256(page).hexdigest()
    rows = [{'url': 'https://a.example/1', 'sha256': saved, 'pattern': 'never looked at', 'passage': 'Launched in April 2014 , it grew.'},
            {'url': 'https://a.example/2', 'sha256': '0' * 64, 'pattern': r'launched in april \d{4}', 'passage': 'it grew.'},
            {'url': 'https://a.example/2', 'sha256': '0' * 64, 'pattern': 'June 2019', 'passage': 'June 2019'},
            {'url': 'https://a.example/3', 'sha256': saved, 'pattern': 'x', 'passage': 'x'},
            {'url': 'https://a.example/4.pdf', 'sha256': '0' * 64, 'pattern': 'April 2014', 'passage': 'April 2014'}]
    asked = []

    def get(url):
        asked.append(url)
        if url.endswith('/3'):
            raise OSError('timed out')
        return (b'%PDF-1.7 April 2014', 'application/pdf') if url.endswith('.pdf') else (page, 'text/html; charset=utf-8')

    check_receipts.check(rows, pause=0, get=get, say=lambda line: None)
    assert [row['today'] for row in rows] == ['same bytes', 'still matches', 'no longer matches', 'not fetched', 'a PDF, not read']
    # the shown passage is tested on its own: a script's text is never part of a page's text
    assert [row['passage_today'] for row in rows] == ['on the page', 'on the page', 'not on the page', 'unknown', 'unknown']
    assert asked == [f'https://a.example/{name}' for name in ('1', '2', '3', '4.pdf')]  # one request per source
    for url in ('file:///etc/hostname', 'http://a.example/1'):
        with pytest.raises(ValueError):
            check_receipts.fetch(url)  # https only


def test_the_rules_with_no_model_and_where_they_fail():
    stronger, simpler = (breakdown.summarise(breakdown.tally(breakdown.rule_answers(ITEMS, rule)['answers'], PROV)[0]) for rule in ('kinds', 'words'))
    close_items = [p for _, _, p in ROWS if p['stratum'] in breakdown.CLOSE]
    decoys = sum(p['decoy'] for p in close_items)
    # The stronger rule is right on every close passage without a decoy and wrong on every one with a decoy.
    close = stronger['rules_without_evidence_on_close_passages']
    assert close['of'] == len(close_items) and close['count'] == decoys > 0
    # The simpler rule, which the close passages were first built against, is wrong on all of them.
    assert simpler['rules_without_evidence_on_close_passages']['count'] == len(close_items)
    assert simpler['score'] < stronger['score']
    # The README states the stronger rule's score.
    assert f"{stronger['score']:.1f}" in (HERE / 'README.md').read_text(encoding='utf-8')


def test_groups_link_items_that_share_a_claim_or_a_passage():
    prov = {'a': {'cluster': 'r1', 'evidence_from': 'r1'}, 'b': {'cluster': 'r2', 'evidence_from': 'r1'},
            'c': {'cluster': 'r3', 'evidence_from': 'r3'}, 'd': {'cluster': 'r4', 'evidence_from': 'r2'}}
    assert breakdown.groups_of(prov) == {'a': 'r1', 'b': 'r1', 'c': 'r3', 'd': 'r1'}
    # two prompts that show the same passage word for word are one group, whatever their claims and records
    prov['e'] = {'cluster': 'r5', 'evidence_from': 'r6'}
    assert breakdown.groups_of(prov, {'c': ['The same words.'], 'e': ['Other words.', 'The same words.']}) == {'a': 'r1', 'b': 'r1', 'c': 'r3', 'd': 'r1', 'e': 'r3'}
    shown = {x['id']: breakdown.passages_of(x['prompt']) for rows in ITEMS.values() for x in rows}
    assert all(len(shown[i]) == len(p['evidence']) for i, p in PROV.items())
    groups = breakdown.groups_of(PROV, shown)
    assert len(set(groups.values())) < len(set(breakdown.groups_of(PROV).values())) < len({p['cluster'] for p in PROV.values()})
    for item_id, p in PROV.items():  # a passage's record and the claim it is shown with are always in one group
        assert groups[item_id] == groups[next(i for i, q in PROV.items() if q['cluster'] == p['cluster'])]
    seen = {}
    for item_id, texts in shown.items():  # and so are any two items that show one passage
        for text in texts:
            assert groups[seen.setdefault(text, item_id)] == groups[item_id]




def test_two_runs_are_compared_item_by_item():
    assert breakdown.sign_test(0, 0) == 1 and breakdown.sign_test(5, 5) == 1 and breakdown.sign_test(10, 0) == pytest.approx(2 / 2 ** 10)
    assert breakdown.sign_test(11, 5) == pytest.approx(0.2101, abs=1e-4) == breakdown.sign_test(5, 11)

    def record(i, family, gold, said, stratum='x'):
        return {'id': f'{family}{i}', 'family': family, 'cluster': f'c{i}', 'gold': gold, 'said': said, 'right': said == gold, 'stratum': stratum}
    # The other run leaves one supported item (3) and one insufficient item (2) unanswered: None is "no letter".
    mine = [record(i, 'ev_supported', 'supported', 'supported') for i in range(4)]
    theirs = [record(i, 'ev_supported', 'supported', ['supported', 'supported', 'insufficient', None][i]) for i in range(4)]
    for i in range(4):  # on a passage that settles nothing, this run rules three times and the other once
        mine.append(record(i, 'ev_insufficient', 'insufficient', 'refuted' if i < 3 else 'insufficient', 'related' if i < 2 else 'cross_page'))
        theirs.append(record(i, 'ev_insufficient', 'insufficient', ['supported', 'insufficient', None, 'insufficient'][i], 'related' if i < 2 else 'cross_page'))
    theirs.append(record(9, 'ev_refuted', 'refuted', 'refuted'))  # an item only one run answered is left out
    result = breakdown.versus(mine, theirs)
    assert result['items_in_both_runs'] == 8 and (result['score'], result['other_score'], result['difference']) == (62.5, 50.0, 12.5)
    assert result['difference_by_family'] == {'ev_supported': 50.0, 'ev_insufficient': -25.0}
    low, high = result['difference_interval_95']
    assert low <= result['difference'] <= high
    assert result['no_answer'] == {'this_run': 0, 'other_run': 2, 'of': 8} and result['groups'] == 4
    # Each run with its unfinished items answered at its own accuracy in each family: here the other run's two.
    assert result['at_own_rate'] == {'score': 62.5, 'other_score': 66.7, 'difference': -4.2}
    assert result['items_the_other_run_left_unfinished'] == {'items': 2, 'this_run_right': 1, 'rest': 6, 'this_run_right_on_the_rest': 4}
    # Counted on every item, an unanswered item looks like holding back; the sign test counts one vote per group.
    assert result['rules_without_evidence'] == {'this_run': 3, 'other_run': 1, 'of': 4, 'groups': 4, 'only_this_run': 2, 'only_other_run': 0,
                                                'group_votes': [2, 0], 'sign_test_p': 0.5, 'claim_votes': [2, 0], 'sign_test_p_by_claim': 0.5,
                                                'other_run_unfinished': 1, 'other_run_at_own_rate': 1.3}
    close = result['rules_without_evidence_on_close_passages']
    assert (close['of'], close['this_run'], close['other_run']) == (2, 2, 1)
    # The view on the items both answered leaves out every item either run left without a letter.
    both = result['when_both_gave_a_letter']
    assert both['items'] == 6 and (both['score'], both['other_score']) == (66.7, 66.7) and both['difference'] == 0
    assert both['difference_by_family'] == {'ev_supported': 33.3, 'ev_insufficient': -33.3}
    assert both['rules_without_evidence'] == {'this_run': 2, 'other_run': 1, 'of': 3, 'groups': 3, 'only_this_run': 1, 'only_other_run': 0,
                                              'group_votes': [1, 0], 'sign_test_p': 1.0, 'claim_votes': [1, 0], 'sign_test_p_by_claim': 1.0}
    assert both['declines_a_clear_case'] == {'this_run': 0, 'other_run': 1, 'of': 3}
    # Two items of one group lean the same way: one vote, not two.
    mine[5]['cluster'] = mine[6]['cluster'] = theirs[5]['cluster'] = theirs[6]['cluster'] = 'shared'
    assert breakdown.versus(mine, theirs)['rules_without_evidence']['group_votes'] == [1, 0]
    # A group can hold several claims: one vote per group, while each claim still votes on its own.
    for x in (mine[5], theirs[5]):
        x['group'], x['cluster'] = 'g', 'c5'
    for x in (mine[6], theirs[6]):
        x['group'], x['cluster'] = 'g', 'c6'
    linked = breakdown.versus(mine, theirs)['rules_without_evidence']
    assert linked['group_votes'] == [1, 0] and linked['claim_votes'] == [2, 0]
