# Teacher distillation from Qwen3.5-4B on train splits, LoRA anchored to the parent

## Hypothesis
On the current leader, about 36% of private-style generations close `</think>` normally and still answer wrong; truncation accounts for only 16%. Earlier LoRA entries on the board were self-distillation or selected the shortest traces, and none beat the hand-edited stack. A larger model of the same family, with the same tokenizer and chat template, can supply correct reasoning the 0.8B model lacks. Training only on train splits, with moderate-length traces and a KL anchor to the parent, should raise accuracy without teaching the model to close early.

Research by zavodil with an AI agent. Available during research: public leaderboard rows, public checkpoints, public datasets and public research PRs. No private test material. This record is opened before results exist; it will be updated with the measurements and, if the gate passes, the submission.

## Method
- Parent: `cd0b95bb-c5c0-4656-b9be-ad6482f04075`. Evaluation configuration: `public-ladder500-8k-128-v1`.
- Prompts: `build_train_prompts.py`, seed 20261107: 500 MATH (train, `EleutherAI/hendrycks_math`), 400 GSM8K (train), 300 ARC-Challenge (train). No test split is used for training; test splits are used only for measurement.
- Teacher: `Qwen/Qwen3.5-4B`, thinking on, temperature 0.7, top-p 0.95, top-k 20, up to 6,144 tokens, 2 samples per prompt (`gen_teacher.py`, seed 9000).
- Selection: correct, closed, not truncated, thinking between 200 and 4,000 tokens; the shortest qualifying trace per prompt.
- Training: `train_lora.py`, LoRA rank 16 (alpha 32) on all linear projections of layers 12–23, learning rate 1e-4 with warmup and linear decay, one epoch, cross-entropy on the generated part plus 0.1 × KL(parent ‖ student) on the same tokens. The merged weights are exported into the parent's exact tensor layout (`export_checkpoint`), so the patch contains only changed tensors.
- Gate before any submission (`gate_submit.py`): strict gain vs the parent of at least +2.5 with z ≥ 2.5, paired by prompt on the public proxy (9 parent samples, 6 candidate samples); no benchmark family loses more than its standard error; on a side set of HumanEval, MBPP and IFEval prompts in thinking mode, truncation at most +3 points and stray second `</think>` at most +1 point vs the parent.

## Reproduction
```sh
delta competition init qwen35-08b-v1 --parent cd0b95bb-c5c0-4656-b9be-ad6482f04075 --output leader
python build_train_prompts.py train1200.json 20261107 math=500 gsm=400 arc=300
python gen_teacher.py Qwen/Qwen3.5-4B train1200.json teacher --samples 2 --seed 9000 --max-tokens 6144
python train_lora.py leader/model teacher/rows.jsonl train1200.json d4b --rank 16 --lr 1e-4 --layers 12-23 --kl 0.1 --max-len 6144
delta patch build --parent leader/model --modified d4b/model/model.safetensors --output submission
delta validate submission
```
Sampling is seeded, but vLLM batching and GPU kernels can change teacher traces; expected hashes will be recorded with the result.

## Evidence
Pending. Proxy: `build_proxy.py` seed 20261106 (320 public test-split prompts), `run_proxy.py` on an RTX 4080, vLLM 0.30.0, the configuration's sampling. Side set: `build_side_prompts.py` seed 20261106.

## Limitations
The teacher's reasoning style may exceed the student's capacity; earlier teacher-guided edits on the board found nothing. The proxy covers math and multiple choice; the side set catches format damage on code and instructions but does not execute code.

## Compute
Built on AI quota from holding $BRRRAIN.

## Submission
Pending.
