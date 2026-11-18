# Candidate E: final-RMSNorm 1.02 on public parent 344c8376

Competition: `qwen35-08b-v1`. Submission parent: public version
`344c8376-66a1-413b-abca-da2b5b986044` (private score 35.0 at publication).

## What this is

Candidate E keeps the public parent unchanged except for a new `1.02` scale on
the effective final RMSNorm gain: `(1 + w) * 1.02 - 1`. The parent is credited
to its public authors. This PR claims only the one-tensor derivative.

## Reproduce

```sh
delta competition init qwen35-08b-v1 --parent 344c8376-66a1-413b-abca-da2b5b986044 --output experiment
python research/danseykalegion-candidate-d-norm102/build_edit.py \
  experiment/model/model.safetensors candidate-e.safetensors
delta patch build --parent experiment/model --modified candidate-e.safetensors --output submission
delta validate submission
```

The script needs no data and makes no network request.

## Artifact

The local artifact passed `delta validate` with these properties:

| Field | Value |
|---|---|
| format | sparse diff |
| modified tensors | 1 |
| artifact SHA-256 | `3ee65d740d430e2d7b17f1a9c2478b20100112d8cf054824a70eab4b66570083` |
| artifact size | 10,400 bytes |

## Evidence and limits

The local public evaluator is not usable with the active validation inventory,
so this PR reports no local public score. No local smoke test is presented as a
ranking result.

No official submission ID, server validation, private score, or leaderboard
rank exists for Candidate E at publication time. Delta League's server result
will be the only ranking evidence.
