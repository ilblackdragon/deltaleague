import asyncio
import json
import sys
import tarfile
import types
from pathlib import Path
from delta_cli.public_eval.native_trial import pin_simple_docker_images


def test_build_context_bytes_and_immutable_image_launch(tmp_path,monkeypatch):
    class Environment:
        async def start(self,force_build):
            assert not force_build
            assert self.task_env_config.docker_image==self._env_vars.prebuilt_image_name
            return self.task_env_config.docker_image
    monkeypatch.setitem(sys.modules,'harbor.environments.docker.docker',types.SimpleNamespace(DockerEnvironment=Environment))
    monkeypatch.setenv('DELTA_NATIVE_BUILD_CACHE',str(tmp_path/'cache'))
    built=[]
    def run(command,**kwargs):
        if command[:3]==['docker','image','inspect']:return types.SimpleNamespace(returncode=0)
        assert command[:2]==['docker','build'] and command[-1]=='-'
        stream=kwargs['stdin'];assert stream.read(2)==b'\x1f\x8b';stream.seek(0)
        with tarfile.open(fileobj=stream,mode='r:gz') as archive:built.append(archive.extractfile('./identity').read().decode())
        Path(command[command.index('--iidfile')+1]).write_text('sha256:'+built[-1])
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr('subprocess.run',run)
    pin_simple_docker_images()
    async def check():
        for identity in ('aaaa','bbbb','aaaa'):
            directory=tmp_path/identity;directory.mkdir(exist_ok=True)
            (directory/'identity').write_text(identity)
            env=Environment();env.environment_dir=directory;env.environment_id=identity
            env.task_env_config=types.SimpleNamespace(docker_image=None)
            env._env_vars=types.SimpleNamespace(prebuilt_image_name=None)
            env._environment_docker_compose_path=directory/'docker-compose.yaml'
            env.extra_docker_compose_paths=[]
            assert await env.start(True)=='sha256:'+identity
    asyncio.run(check())
    assert built==['aaaa','bbbb']
