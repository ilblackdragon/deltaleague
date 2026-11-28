"""Serve a checkpoint on the local ROCm host with the pinned evaluation context."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

IMAGE = ("docker.io/kyuz0/vllm-therock-gfx1151@"
         "sha256:f8a7433d662b37b200ca19996d4013288071341ad514e66de69ea4871cfa3966")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sequences", type=int, default=128)
    args = parser.parse_args()
    context = json.loads(Path("evaluation.json").read_text())["generation"]["max_model_len"]
    subprocess.run(
        [
            "podman", "run", "-d", "--rm", "--name", "dl-r2-vllm",
            "--device", "/dev/kfd", "--device", "/dev/dri",
            "--group-add", "keep-groups", "--security-opt", "seccomp=unconfined", "--ipc=host",
            "-p", "127.0.0.1:18000:8000", "-v", f"{args.checkpoint.resolve()}:/model:ro",
            "-v", f"{Path.home()}/.cache/vllm-container:/cache",
            "-e", "TRITON_CACHE_DIR=/cache/triton", "-e", "VLLM_CACHE_ROOT=/cache/vllm",
            "-e", "VLLM_ROCM_USE_AITER=0", "-e", "VLLM_ROCM_USE_AITER_LINEAR=0",
            IMAGE, "vllm", "serve", "/model", "--served-model-name", "policy",
            "--host", "0.0.0.0", "--port", "8000", "--language-model-only", "--dtype", "bfloat16",
            "--max-model-len", str(context), "--max-num-seqs", str(args.sequences),
            "--gpu-memory-utilization", "0.5", "--attention-backend", "TRITON_ATTN", "--no-enable-log-requests",
        ],
        check=True,
    )
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        try:
            with urlopen("http://127.0.0.1:18000/v1/models", timeout=5) as response:
                model, = json.load(response)["data"]
            if model.get("max_model_len") != context:
                raise RuntimeError(f"Endpoint context {model.get('max_model_len')} differs from {context}")
            print(f"Ready: {args.checkpoint} with context {context}", flush=True)
            return
        except OSError:
            # During startup Podman's forwarder can reset an accepted socket.
            # Keep polling while the container is alive.
            if subprocess.run(["podman", "container", "exists", "dl-r2-vllm"]).returncode != 0:
                raise RuntimeError("vLLM container exited before becoming ready")
            time.sleep(5)
    raise TimeoutError("vLLM did not become ready")


if __name__ == "__main__":
    main()
