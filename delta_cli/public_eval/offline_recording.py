"""Supply Harbor's optional recording tool without sandbox network access.

Keep existing installed tools and Harbor's original fallback behavior. The wheel
is distributed unmodified with its upstream license and verified before upload.
"""
import hashlib
from pathlib import Path

WHEEL='asciinema-2.4.0-py3-none-any.whl'
SHA256='249ccf108509d643ddaf38979dac48c99e9e251f597173d8553a30d7a6423104'


async def ensure_recording_tool(environment):
    existing=await environment.exec(command='asciinema --version',user='root')
    if existing.return_code==0:return True
    wheel=Path(__file__).parent/'data'/WHEEL
    if hashlib.sha256(wheel.read_bytes()).hexdigest()!=SHA256:
        raise ValueError('Offline recording wheel integrity mismatch')
    target='/tmp/'+WHEEL
    await environment.upload_file(source_path=wheel,target_path=target)
    result=await environment.exec(command='python3 -m pip install --no-index --no-deps --disable-pip-version-check '+target,
        user='root',timeout_sec=30)
    return result.return_code==0


def enable_offline_recording():
    from harbor.agents.terminus_2.terminus_2 import TmuxSession
    if getattr(TmuxSession,'_delta_offline_recording',False):return
    original=TmuxSession._attempt_tmux_installation
    async def install(self):
        if self._remote_asciinema_recording_path:
            try:await ensure_recording_tool(self.environment)
            except (OSError,RuntimeError,TimeoutError):
                # Other native environments may lack Python/pip or upload support.
                # Keep Harbor's own installation/fallback policy in those cases.
                pass
        return await original(self)
    TmuxSession._attempt_tmux_installation=install
    TmuxSession._delta_offline_recording=True
