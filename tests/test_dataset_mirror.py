import hashlib
import io
import tarfile
from unittest.mock import patch
import pytest
from delta_cli.public_eval.dataset_mirror import prepare


def archive(name='README.md'):
    body=io.BytesIO()
    with tarfile.open(fileobj=body,mode='w:gz') as tar:
        info=tarfile.TarInfo(name);info.size=4;tar.addfile(info,io.BytesIO(b'data'))
    return body.getvalue()


class Response:
    def __init__(self,body):self.body=body
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def raise_for_status(self):pass
    def iter_bytes(self):yield self.body


def test_mirror_verified_once_then_reused(tmp_path):
    body=archive();ref={'url':'https://example.test/data','sha256':hashlib.sha256(body).hexdigest(),'revision':'pinned'}
    with patch('httpx.stream',return_value=Response(body)) as download:
        result=prepare({'fixture':ref},tmp_path)
        assert prepare({'fixture':ref},tmp_path)==result
        assert download.call_count==1
    assert result['fixture']['revision']=='pinned'


@pytest.mark.parametrize('bad', ['digest','path'])
def test_mirror_rejects_bad_sources(tmp_path,bad):
    body=archive('../escape' if bad=='path' else 'data.json')
    ref={'url':'https://example.test/data','sha256':'0'*64 if bad=='digest' else hashlib.sha256(body).hexdigest(),'revision':'pinned'}
    with patch('httpx.stream',return_value=Response(body)),pytest.raises(ValueError):prepare({'fixture':ref},tmp_path)
    assert not (tmp_path.parent/'escape').exists()
