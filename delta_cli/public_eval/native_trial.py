"""Run each native Harbor trial in its own interpreter.

Harbor owns event-loop-bound global state. Separate interpreters let callers
schedule concurrent trials without sharing that state across asyncio loops.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


def pin_simple_docker_images():
    """Build simple task images once, then run their immutable image IDs.

    Compose builds of same-named services have mixed local contexts under
    concurrent trials. A host-shared lock protects builds; running containers
    and model inference remain concurrent. No Compose rebuild occurs on up.
    """
    import asyncio, fcntl, tempfile, tarfile
    from harbor.environments.docker.docker import DockerEnvironment
    if getattr(DockerEnvironment,'_delta_image_identity',False):return
    original=DockerEnvironment.start
    def build(self):
        cache=Path(os.environ.get('DELTA_NATIVE_BUILD_CACHE',str(Path(tempfile.gettempdir())/'delta-native-images')))
        cache.mkdir(parents=True,exist_ok=True)
        key='tar-context-v2-'+self.environment_id
        receipt=cache/(key+'.json')
        with (cache/'build.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if receipt.exists():
                image=json.loads(receipt.read_text())['image']
                if subprocess.run(['docker','image','inspect',image],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:return image
            with tempfile.TemporaryDirectory(dir=cache) as tmp:
                iid=Path(tmp)/'image-id'
                with tempfile.TemporaryFile() as context:
                    with tarfile.open(fileobj=context,mode='w:gz',format=tarfile.GNU_FORMAT) as archive:
                        archive.add(self.environment_dir,arcname='.')
                    context.seek(0)
                    subprocess.run(['docker','build','--iidfile',str(iid),'-'],stdin=context,check=True)
                image=iid.read_text().strip()
                if not image.startswith('sha256:'):raise RuntimeError('Missing immutable image ID')
                receipt.write_text(json.dumps({'context_sha256':key,'image':image}))
                return image
    async def start(self,force_build):
        if not self.task_env_config.docker_image and not self._environment_docker_compose_path.exists() and not self.extra_docker_compose_paths:
            self.task_env_config.docker_image=await asyncio.to_thread(build,self)
            self._env_vars.prebuilt_image_name=self.task_env_config.docker_image
            force_build=False
        return await original(self,force_build)
    DockerEnvironment.start=start
    DockerEnvironment._delta_image_identity=True


def run_trial_isolated(config, *, tokenizer_path=None):
    from harbor.models.trial.result import TrialResult
    root=Path(config.trials_dir)
    root.mkdir(parents=True,exist_ok=True)
    request=root/(config.trial_name+'.launch.json')
    result=root/(config.trial_name+'.native.json')
    if request.exists() or result.exists():
        raise FileExistsError('Use a unique trial name for each execution attempt')
    payload={'config':config.model_dump(mode='json'),'tokenizer_path':str(tokenizer_path) if tokenizer_path else None}
    with request.open('x') as f:json.dump(payload,f)
    request.chmod(0o600)
    with (root/(config.trial_name+'.process.log')).open('x') as log:
        process=subprocess.run([sys.executable,str(Path(__file__).resolve()),str(request),str(result)],stdout=log,stderr=subprocess.STDOUT)
    if process.returncode or not result.is_file():raise RuntimeError('Native trial subprocess failed; inspect process log')
    return TrialResult.model_validate_json(result.read_text())


def main():
    import asyncio
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial
    pin_simple_docker_images()
    payload=json.loads(Path(sys.argv[1]).read_text())
    if payload.get('tokenizer_path'):
        from transformers import AutoTokenizer
        from harbor.agents.terminus_2.terminus_2 import Terminus2
        from delta_cli.public_eval.terminal import context_tokens
        tokenizer=AutoTokenizer.from_pretrained(payload['tokenizer_path'],trust_remote_code=False)
        Terminus2._count_total_tokens=lambda self,chat:context_tokens(tokenizer,chat.messages)
    config=TrialConfig.model_validate(payload['config'])
    async def go():return await (await Trial.create(config)).run()
    result=asyncio.run(go())
    Path(sys.argv[2]).write_text(result.model_dump_json())

if __name__=='__main__':
    os.umask(0o077)
    main()
