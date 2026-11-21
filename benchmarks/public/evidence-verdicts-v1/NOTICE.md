# Notice: sources, licences and attribution

Everything in this folder was public before it was put here. This file says where each part comes from and on what terms it is offered. The repository has no licence file at the time of writing, so each part names its own; a maintainer's decision on how benchmark data here is licensed replaces the defaults below wherever it can.

## Wikipedia sentences and the claims made from them

`records.json` holds 301 sentences (`text`) from 20 English Wikipedia articles, each at the revision below, under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Attribution goes to each article's contributors; the page history is at `https://en.wikipedia.org/w/index.php?title=<title>&action=history`, and every record and every item's entry in `provenance.json` carries the revision URL. The short checkable claims (`claim`) and the edited claims of `ev_refuted` are adaptations of those sentences and are offered under the same licence. An edited claim is false on purpose: its entry in `provenance.json` has `"modified": true` and gives the original.

| Article | Revision | Revision time (UTC) | Records |
|---|---|---|---|
| [Aave](https://en.wikipedia.org/w/index.php?title=Aave&oldid=1383227531) | 1383227531 | 2026-11-01T17:21:21Z | 7 |
| [Avalanche (blockchain platform)](https://en.wikipedia.org/w/index.php?title=Avalanche_(blockchain_platform)&oldid=1381043012) | 1381043012 | 2026-10-19T17:15:07Z | 27 |
| [Bitcoin](https://en.wikipedia.org/w/index.php?title=Bitcoin&oldid=1382376340) | 1382376340 | 2026-10-28T11:17:12Z | 35 |
| [Cardano (blockchain platform)](https://en.wikipedia.org/w/index.php?title=Cardano_(blockchain_platform)&oldid=1345627574) | 1345627574 | 2026-03-27T04:21:32Z | 14 |
| [Chainlink (blockchain oracle)](https://en.wikipedia.org/w/index.php?title=Chainlink_(blockchain_oracle)&oldid=1373689437) | 1373689437 | 2026-09-07T09:49:59Z | 8 |
| [Dai (cryptocurrency)](https://en.wikipedia.org/w/index.php?title=Dai_(cryptocurrency)&oldid=1375647492) | 1375647492 | 2026-09-19T03:29:12Z | 6 |
| [Decentralized finance](https://en.wikipedia.org/w/index.php?title=Decentralized_finance&oldid=1382583755) | 1382583755 | 2026-10-29T10:17:46Z | 6 |
| [Ethereum](https://en.wikipedia.org/w/index.php?title=Ethereum&oldid=1383352660) | 1383352660 | 2026-11-02T07:03:53Z | 65 |
| [Monero](https://en.wikipedia.org/w/index.php?title=Monero&oldid=1383327494) | 1383327494 | 2026-11-02T03:23:26Z | 15 |
| [NEAR (blockchain platform)](https://en.wikipedia.org/w/index.php?title=NEAR_(blockchain_platform)&oldid=1374559820) | 1374559820 | 2026-09-12T20:05:09Z | 3 |
| [Polkadot (blockchain platform)](https://en.wikipedia.org/w/index.php?title=Polkadot_(blockchain_platform)&oldid=1378697257) | 1378697257 | 2026-10-05T18:55:58Z | 13 |
| [Polymarket](https://en.wikipedia.org/w/index.php?title=Polymarket&oldid=1383536551) | 1383536551 | 2026-11-03T05:52:06Z | 7 |
| [Prediction market](https://en.wikipedia.org/w/index.php?title=Prediction_market&oldid=1383537451) | 1383537451 | 2026-11-03T05:59:38Z | 20 |
| [Proof of stake](https://en.wikipedia.org/w/index.php?title=Proof_of_stake&oldid=1373286584) | 1373286584 | 2026-09-05T02:25:36Z | 5 |
| [Solana (blockchain platform)](https://en.wikipedia.org/w/index.php?title=Solana_(blockchain_platform)&oldid=1370327076) | 1370327076 | 2026-08-20T13:48:00Z | 3 |
| [Stablecoin](https://en.wikipedia.org/w/index.php?title=Stablecoin&oldid=1383799546) | 1383799546 | 2026-11-04T23:08:36Z | 13 |
| [Tether (cryptocurrency)](https://en.wikipedia.org/w/index.php?title=Tether_(cryptocurrency)&oldid=1382852995) | 1382852995 | 2026-10-30T17:27:49Z | 21 |
| [USDC (cryptocurrency)](https://en.wikipedia.org/w/index.php?title=USDC_(cryptocurrency)&oldid=1370061318) | 1370061318 | 2026-08-18T22:35:20Z | 7 |
| [Uniswap](https://en.wikipedia.org/w/index.php?title=Uniswap&oldid=1379919424) | 1379919424 | 2026-10-12T17:50:08Z | 18 |
| [Zcash](https://en.wikipedia.org/w/index.php?title=Zcash&oldid=1381362683) | 1381362683 | 2026-10-21T22:37:57Z | 8 |

## Passages from sources

A passage (the `excerpt` of a record's receipt or of its `near` close passage, and the same text inside the prompts) is a quotation of at most 300 characters from the public page at its `url`, kept with its fetch time and the SHA-256 of what was fetched. 251 passages from 151 pages on 66 hosts are shown in prompts. The passages belong to their publishers and are quoted so that a claim can be checked against them; no page is reproduced. `records.json` keeps passage text only where a prompt can show it: receipts on news publishers' pages (`excluded_hosts` in `config.json`) and API, RPC or data-file responses keep their URL, hash and the short pattern that matched (at most 66 characters), and no passage; a receipt that points to a JSON file saved with the corpus keeps its path and pattern. If a source's owner wants a passage gone, removing its record and running `build_set.py` gives a new set and a new seal.

## Labels, rules and reports

`provenance.json`, `report.json`, `dropped.json`, `vetoed.json`, `handcheck.json`, `solvers.json`, `baselines/` and `config.json` were made for this proposal. They are offered under CC BY-SA 4.0, like the claims they describe. `solvers.json` and `baselines/` hold letters, scores, finish reasons and token counts only; no model's generated text is in this folder.

## Code

`build_set.py`, `near_miss.py`, `ask_models.py`, `hand_check.py`, `check_receipts.py` and `breakdown.py` are offered under the MIT licence below. `tests/test_evidence_verdicts.py` sits in the repository's own `tests/` folder: it is offered under whatever licence the maintainer gives that folder, and under the MIT licence until then.

> Copyright (c) 2026 the Priced In authors
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## People

No claim names a person. The builder refuses a claim, and a close passage, that matches the `people` pattern in `config.json`, which is a list of names, not a classifier; every claim in the set was also read for names before release. The other passages are quoted as their sources wrote them, and some name people in their public roles: founders, officials and authors.

## Who made what

The records were annotated by AI models (Claude Opus 5.5 and Claude Fable 5.2) under a written method, and the builder, tools and tests were written with Claude Code. One person, the proposal's author, read 30 items blind. The README says what was and was not checked by a person.
