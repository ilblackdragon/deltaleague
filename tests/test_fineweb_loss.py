"""FineWeb validation loss: framing, reduction, sharding and the standalone path."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading

import pytest

from delta_cli.public_eval import fineweb_loss, standalone

EOT = 1


class CharTokenizer:
    """One token per character; ids start at 2 so 1 can stand for <|endoftext|>."""
    unk_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [2 + ord(c) % 50 for c in text]}

    def convert_tokens_to_ids(self, token):
        return EOT if token == '<|endoftext|>' else 0


DOCS = [{'i': 0, 'text': 'abcde', 'gpt2_tokens': 3, 'predicts_eot': True, 'bytes': 5},
        {'i': 1, 'text': 'xy', 'gpt2_tokens': 1, 'predicts_eot': True, 'bytes': 2},
        {'i': 2, 'text': 'tail', 'gpt2_tokens': 2, 'predicts_eot': False, 'bytes': 4}]


def test_windows_cover_every_target_once_and_restart_from_eot():
    doc = {'text': 'abcdefg', 'predicts_eot': True}
    prompts = fineweb_loss.windows(doc, CharTokenizer(), EOT, size=4)
    assert all(p[0] == EOT and len(p) <= 4 for p in prompts)
    targets = [t for p in prompts for t in p[1:]]
    assert targets == CharTokenizer()('abcdefg')['input_ids'] + [EOT]
    tail = fineweb_loss.windows({'text': 'ab', 'predicts_eot': False}, CharTokenizer(), EOT)
    assert tail == [[EOT] + CharTokenizer()('ab')['input_ids']]


def native(prompt, value=-0.5):
    entries = [None] + [{str(t): {'logprob': value, 'rank': 1, 'decoded_token': 'x'}} for t in prompt[1:]]
    return {'choices': [{'index': 0, 'text': 'z', 'finish_reason': 'length', 'prompt_logprobs': entries}],
            'usage': {'prompt_tokens': len(prompt), 'completion_tokens': 1}}


def test_reduction_is_compact_and_matches_native_sum():
    prompt = [EOT, 5, 6, 7]
    request = {'prompt': prompt, 'delta_reduce': fineweb_loss.REDUCE}
    reduced = fineweb_loss.reduce_response(request, native(prompt))
    choice = reduced['choices'][0]
    assert choice['prompt_logprobs'] is None and choice['delta_prompt_logprob_count'] == 3
    assert fineweb_loss.logprob_sum(prompt, reduced) == fineweb_loss.logprob_sum(prompt, native(prompt)) == -1.5
    assert fineweb_loss.reduce_response({'prompt': prompt}, native(prompt)) == native(prompt)
    with pytest.raises(ValueError):
        fineweb_loss.logprob_sum(prompt + [8], reduced)
    with pytest.raises(ValueError):
        fineweb_loss.logprob_sum(prompt, {'choices': [{'text': 'z'}]})


def test_shard_totals_combine_exactly():
    a = {'totals': {'nll_sum_nats': 10.0, 'candidate_targets': 4, 'gpt2_targets': 5, 'utf8_bytes': 8, 'windows': 2, 'split_documents': 0}}
    b = {'totals': {'nll_sum_nats': 2.0, 'candidate_targets': 2, 'gpt2_targets': 1, 'utf8_bytes': 4, 'windows': 1, 'split_documents': 1}}
    combined = fineweb_loss.combine([a, b])
    assert combined['value'] == combined['loss_per_gpt2_token'] == 2.0
    assert combined['nats_per_token'] == 2.0
    assert math.isclose(combined['bits_per_byte'], 12 / math.log(2) / 12)
    assert combined['lower_is_better'] and combined['totals']['split_documents'] == 1


def test_protocol_declares_pinned_window_without_generation():
    p = standalone.make_protocol(standalone.FINEWEB, {}, coverage='full', concurrency=16)
    assert p['expected_items'] == fineweb_loss.DOCUMENTS == 14983
    assert p['documents_sha256'] == fineweb_loss.DOCUMENTS_SHA256 and p['metric_kind'] == 'loss'
    assert 'generation' not in p and 'sampling' not in p
    with pytest.raises(ValueError):
        standalone.make_protocol(standalone.FINEWEB, {}, coverage='sample')
    with pytest.raises(ValueError):
        standalone.make_protocol(standalone.FINEWEB, {}, coverage='full', max_model_len=fineweb_loss.SCORING_WINDOW)


@pytest.fixture
def upstream():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(request)
            body = json.dumps(native(request['prompt'])).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{server.server_port}/v1', seen
    server.shutdown()
    server.server_close()


def test_standalone_shard_scores_through_the_caching_proxy(tmp_path, monkeypatch, upstream):
    endpoint, seen = upstream
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', staticmethod(lambda *a, **k: CharTokenizer()))
    monkeypatch.setattr(fineweb_loss, 'load_documents', lambda bounds=None: DOCS[bounds[0]:bounds[1]] if bounds else DOCS)
    tokenizer = tmp_path / 'tokenizer'
    tokenizer.mkdir()
    protocol = standalone.make_protocol(standalone.FINEWEB, {}, coverage='full', concurrency=2)
    result = standalone.run_benchmark(protocol=protocol, endpoint=endpoint, model='m', tokenizer=tokenizer,
                                      content_hash='a' * 64, output=tmp_path / 'out', bounds=[0, 3])
    metrics = result[standalone.FINEWEB]['metrics']
    totals = metrics['totals']
    # Targets: 5+1, 2+1 and 4 (truncated tail predicts no <eot>), each at -0.5 nats.
    assert totals['candidate_targets'] == 13 and totals['gpt2_targets'] == 4 + 2 + 2
    assert math.isclose(totals['nll_sum_nats'], 6.5)
    assert math.isclose(metrics['value'], 6.5 / 8) and metrics['lower_is_better']
    assert result[standalone.FINEWEB]['completed_items'] == 3
    assert {r['delta_reduce'] for r in seen} == {fineweb_loss.REDUCE}
    assert all(r['max_tokens'] == 1 and r['prompt_logprobs'] == 0 for r in seen)
    assert (tmp_path / 'out' / (standalone.FINEWEB + '.report.json.gz')).exists()
