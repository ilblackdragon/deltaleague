import pytest

from delta_cli.scoring import score_answer


@pytest.mark.parametrize('gold, prediction, separators', [
    ('red,blue', 'red, blue', ','),
    ('a=3;b=7', 'a = 3 ; b = 7', ';='),
    ('cat|dog', 'cat | dog', '|'),
    ('2/9', '2 / 9', '/'),
    ('NONE', 'none', ','),
    ('4,5', '<think>draft</think>4,\n5', ','),
])
def test_opt_in_separator_whitespace(gold, prediction, separators):
    assert score_answer({'scorer': 'exact_delimited', 'answer': gold,
                         'answer_separators': separators}, prediction) == 1


@pytest.mark.parametrize('prediction', ['blue,red', 'red,,blue', 'red;blue',
                                      'red,blue,', 'red,blue because', 'r ed,blue'])
def test_delimiters_do_not_relax_content_or_order(prediction):
    assert score_answer({'scorer': 'exact_delimited', 'answer': 'red,blue',
                         'answer_separators': ','}, prediction) == 0


def test_legacy_exact_remains_strict_and_undeclared_punctuation_is_unchanged():
    assert score_answer({'answer': 'red,blue'}, 'red, blue') == 0
    assert score_answer({'scorer': 'exact_delimited', 'answer': 'a=3;b=7',
                         'answer_separators': ';'}, 'a = 3; b=7') == 0
    assert score_answer({'scorer': 'exact_delimited', 'answer': '2/9',
                         'answer_separators': '/'}, '4/18') == 0


@pytest.mark.parametrize('separators', [None, '', [';'], 'x', ',.*'])
def test_invalid_contract_is_rejected(separators):
    with pytest.raises(ValueError, match='answer_separators'):
        score_answer({'scorer': 'exact_delimited', 'answer': 'a',
                      'answer_separators': separators}, 'a')


def test_local_validation_accepts_contract_and_rejects_bad_metadata(workspace):
    import json
    from delta_cli import local
    from delta_cli.artifacts import file_hash
    root, _, config = workspace
    path = root / 'validation.json'
    data = json.loads(path.read_text())
    data['a'][0].update(scorer='exact_delimited', answer='red,blue', answer_separators=',')
    path.write_text(json.dumps(data))
    config['validation_sha256'] = file_hash(path)
    tasks, _, _ = local.public_examples(root, config)
    assert score_answer(tasks[0]['example'], 'red, blue') == 1
    data['a'][0]['answer_separators'] = 'letters'
    path.write_text(json.dumps(data))
    config['validation_sha256'] = file_hash(path)
    with pytest.raises(ValueError, match='answer_separators'):
        local.public_examples(root, config)
