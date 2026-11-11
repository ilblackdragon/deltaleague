"""Verified local sources for native datasets; usable without DeltaLeague services."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
from unittest.mock import patch


def prepare(sources, cache):
    import httpx
    from filelock import FileLock
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    paths={}
    for name,ref in sources.items():
        sha=ref['sha256']
        if len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):raise ValueError('Dataset digest')
        target=cache/sha
        with FileLock(str(target)+'.lock',timeout=-1):
            if not (target/'READY').exists():
                with tempfile.TemporaryDirectory(dir=cache) as temp:
                    temp=Path(temp);archive=temp/'source.tar.gz';digest=hashlib.sha256()
                    with httpx.stream('GET',ref['url'],timeout=600) as response:
                        response.raise_for_status()
                        with archive.open('wb') as output:
                            for chunk in response.iter_bytes():digest.update(chunk);output.write(chunk)
                    if digest.hexdigest()!=sha:raise ValueError('Dataset integrity')
                    extracted=temp/'data';extracted.mkdir()
                    with tarfile.open(archive) as tar:
                        for member in tar.getmembers():
                            if member.issym() or member.islnk() or not (member.isfile() or member.isdir()) or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                                raise ValueError('Unsafe dataset archive')
                        tar.extractall(extracted,filter='data')
                    (extracted/'READY').write_text(ref['revision'])
                    extracted.rename(target)
        if (target/'READY').read_text()!=ref['revision']:raise ValueError('Dataset revision mismatch')
        paths[name]={'path':str(target),'revision':ref['revision']}
    return paths


@contextmanager
def use_sources(config):
    sources=config.get('dataset_sources',{})
    if not sources:
        yield
        return
    paths=prepare(sources,os.environ.get('DELTA_DATASET_CACHE','/tmp/delta-benchmark-datasets'))
    with patch.dict(os.environ,{'DELTA_DATASET_SOURCES':json.dumps(paths)}):yield
