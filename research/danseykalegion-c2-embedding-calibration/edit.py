"""Build C2 from an initialized Delta League workspace."""

from __future__ import annotations

import hashlib
from pathlib import Path

from safetensors.torch import load_file, save_file


PARENT = Path("model/model.safetensors")
OUTPUT = Path("candidate-c2.safetensors")
EMBEDDINGS = "model.language_model.embed_tokens.weight"
LOOP_TOKEN_IDS = (13784, 3655, 77264, 84485)
THINK_CLOSE_TOKEN_ID = 248069


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if OUTPUT.exists():
    raise FileExistsError(f"Refusing to overwrite {OUTPUT}")

weights = load_file(str(PARENT))
embedding = weights[EMBEDDINGS].clone()
embedding[list(LOOP_TOKEN_IDS)] *= 0.98
embedding[THINK_CLOSE_TOKEN_ID] *= 1.005
weights[EMBEDDINGS] = embedding
save_file(weights, str(OUTPUT))
print(f"parent_sha256={sha256(PARENT)}")
print(f"candidate_sha256={sha256(OUTPUT)}")
