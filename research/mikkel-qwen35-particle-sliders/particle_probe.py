"""Native teacher mechanics: exact OFF and a live shared-bank counterfactual."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from qwen35_sliders.data import digest, write_json
from qwen35_sliders.host import Slider, hidden
from qwen35_sliders.training import load_host
from qwen35_sliders.upstream import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--upstream")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    runner, provenance = load_host(args.workspace)
    prompts = ["What is 17 times 23?", "Explain why water freezes.", "Write a Python function that adds two numbers."]
    inputs = [runner.tokenizer.encode(runner.tokenizer.apply_chat_template(
        [dict(role="user",content=p)],tokenize=False,add_generation_prompt=True,enable_thinking=True),
        add_special_tokens=False) for p in prompts]
    def responses():
        return torch.cat([hidden(runner.model,ids).detach() for ids in inputs])
    with torch.no_grad():
        base = responses()
        slider = Slider(runner.model,load(args.upstream))
        try:
            slider.load_state_dict(load_file(args.teacher),strict=True)
            with slider.scaled(0.):
                off = responses()
            with slider.scaled(1.):
                teacher = responses()
                saved = slider.particles.detach().clone()
                slider.particles.zero_()
                zero_bank = responses()
                slider.particles.copy_(saved)
            if not torch.equal(base,off):
                raise AssertionError("Reloaded teacher OFF must exactly preserve the base")
            effect = float((teacher-zero_bank).square().mean().sqrt())
            shift = float((teacher-base).square().mean().sqrt())
            if not 0 < effect < float("inf") or not 0 < shift < float("inf"):
                raise AssertionError("Native teacher must have finite nonzero bank and adapter effects")
            result = dict(schema="qwen35-native-bank-probe-v1",teacher_sha256=digest(args.teacher),
                base_content_hash=provenance["parent"],probe_count=len(prompts),target_count=len(slider.names),
                bank_shape=list(slider.particles.shape),exact_off=True,bank_zeroing_hidden_rms=effect,
                teacher_vs_off_hidden_rms=shift,runtime=provenance["runtime"],
                note="Generic serialization/causal bank probes; not capability scores or proof of quality improvement.")
        finally:
            slider.close()
    write_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
