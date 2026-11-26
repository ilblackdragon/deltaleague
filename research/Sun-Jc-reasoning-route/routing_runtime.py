"""Load exact versions of the shared experiment helpers."""
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DEFAULT_HELPERS = Path('/Users/jc/.codex/worktrees/leader96-mixture/deltaleague/research/Sun-Jc-leader96-mixture')


def configure(protocol):
    helpers = Path(os.environ.get('DELTA_MIXTURE_STUDY', DEFAULT_HELPERS))
    for name, digest in protocol['pinned_helpers'].items():
        if hashlib.sha256((helpers / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Unexpected shared runner: ' + name)
    sys.path.insert(0, str(helpers))


@contextmanager
def metal_slot(root):
    with (root.parent / '.reasoning-route-metal.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
