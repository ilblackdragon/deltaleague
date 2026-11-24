# Evidence verdicts v1: a public development set from a fact court's checked claims (competition proposal)

A reading task with three answers. An item is a short claim, one to three passages quoted from primary sources, and the question: do the passages show the claim is true, show it is false, or not settle it? 378 items in the `validation.json` format (scorer `choice`), in three families of one label each, sealed by the SHA-256 in `evidence-verdicts-v1.sha256`. It runs locally with `delta evaluate --validation evidence-verdicts-v1.json`.

**This is public development material, not a test.** Every claim comes from a pinned Wikipedia revision and every passage from a public page, so nothing here can rank a submission, and merging it activates no evaluation. Two ways to get hidden items from the same pipeline are under "Hidden items" below.

What it adds to the public suite: BoolQ asks a yes or no question that its passage always answers, and BFCL's irrelevance cases ask a model to call no function. Here the third answer is "the evidence does not settle it", next to real refutations. 67 of those items show a close passage: it shares words with the claim and leaves the claimed value out, and 30 of them hold another number or year of the same kind, about something else. That is the error that costs most when a model acts on documents, ruling on a claim the evidence does not carry, and `breakdown.py` reports it for any run.

## Examples

One item of each of six kinds, chosen by hash among the items with a single passage that names no listed person and was not flagged for the human read, a passage of at most 240 characters first. In a prompt the three answers follow as lettered options in an order that changes from item to item.

- **supported**, the passage states the claim (`sup.aave-0018`). Claim, about "Aave": *GHO is backed by collateral in Aave.* Evidence: aave.com: "Learn More Multi-collateral Backing GHO is backed by multiple different assets supplied into the Aave Protocol. Learn More Build with GHO With as few as 10 lines of code, you can easily integrate GHO payments into your app."
- **refuted**, a true claim with its year changed (`ref.cardano-0008.year-shift.0`). Claim, about "Cardano (blockchain platform)": *the first version of Cardano was released in 2021.* Evidence: roadmap.cardano.org: "Two years, thousands of GitHub commits, and hundreds of hours of study later, the first version of Cardano shipped in September 2017, and the Byron era began."
- **refuted**, a line that was out of date as written (`ref.tether-0030.natural`). Claim, about "Tether (cryptocurrency)": *Tether issues on Algorand, Avalanche, Celo, Ethereum, EOS, Liquid, Near, Polygon, Solana, Bitcoin Cash SLP, Statemint, Tezos, Tron.* Evidence: tether.to: "Currently Supported Asset Protocols: Please note that Tether is no longer issuing or obligated to redeem Tether Tokens on the Kusama, Bitcoin Cash SLP, Omni Layer, EOS and Algorand blockchains. For more information, please see Tether Provides Update on Transition Plan for Legacy Blockchains ."
- **insufficient**, a passage about another subject (`ins.cardano-0041.prediction-market-0209`). Claim, about "Cardano (blockchain platform)": *users can join stake pools.* Evidence: iemweb.biz.uiowa.edu: "It allows students to invest real money ($5.00-$500.00) and to trade in a variety of contracts. You may be familiar with the best-known part of the IEM, the political markets ."
- **insufficient**, a passage from the claim's own source that leaves the value out (`ins.uniswap-0031.near`). Claim, about "Uniswap": *the UNIfication proposal included a burn of 100 million UNI from the treasury.* Evidence: blog.uniswap.org: "UNIfication Blog Topics Search blog / Launch App Search blog / Launch App All Stories #Protocols November 10, 2025 UNIfication #Protocols Below is the UNIfication proposal that was posted to Uniswap governance."
- **insufficient**, a close passage that holds a decoy: another value of the same kind, about something else (`ins.avalanche-0083.near`). Claim, about "Avalanche (blockchain platform)": *an Avalanche proposal reduced the MinConsumptionRate parameter for Primary Network staking rewards to 7.5%.* Evidence: raw.githubusercontent.com: "Given the current network stake, staking behavior, and protocol parameter configurations, the trailing one-year inflation rate is approximately 5.5%."

## Where the items come from

[Priced In](https://isitpriced.com) is a prediction market for facts in which three open models rule on challenged lines. To measure those judges, sentences of 20 English Wikipedia articles were checked one by one against primary sources, to a method written down before any text was fetched ([METHOD.md](https://github.com/iamdflame/priced-in/blob/0fef6ad75ea7565ec8cfa287e3c89a649f603f27/corpus/METHOD.md)). This set turns the checked sentences into items for the judges' own task. The author of this proposal is the author of Priced In; see "What this set is and is not".

## How it was built

1. **Sentences.** 20 articles on crypto networks, stablecoins and prediction markets, each pinned to its newest revision at or before 2026-11-05T00:00:00Z, split into sentences from the rendered HTML. Each record keeps the revision id and URL.
2. **Checked records** (`records.json`, 301 of them). One AI annotator (Claude Opus 5.5) decided 232 sentences that state something checkable today against primary sources in a fixed order (chain data, then the project's own documents, repositories and filings, then aggregators; never a news story alone): 216 still true, 11 out of date, 5 never true. For this set, 69 more sentences about past events were checked the same way on 20 November 2026 (Claude Fable 5.2). A record holds the sentence, a short checkable claim, and its receipts: source URL, fetch time, the SHA-256 of what was fetched, and a pattern that matched in the saved text. No person re-read all 301; step 7 is the only human check.
3. **Passages.** A passage is at most 300 characters of the saved page, cut at sentence ends around the receipt's match. For a true claim, the sentence of the same saved page that shares the most words and numbers with the claim replaces it when it shares more; a stale or wrong claim always keeps its receipt's own passage. A prompt names a source by host only, since a URL path can give the answer away. `records.json` keeps the text of a passage only where a prompt can show it. The 46 receipts no prompt shows keep no passage: 7 on news publishers' pages and 34 API, RPC or data-file responses keep their URL, hash and the short pattern that matched (at most 66 characters), and 5 point to a JSON file saved with the corpus, by its path, with the pattern and no hash.
4. **Three families, one label each**, so no constant answer scores:
   - `ev_supported`: a true claim with its own passages.
   - `ev_refuted`: a line the annotator caught as stale or wrong, with the passage that contradicts it (7 items); or a true claim with one value changed so that its own passages contradict it (72 items). A number gets a new leading digit, at least a quarter away from the true value. A year in a sentence about a past event moves by 2 to 7. An edit is made only where the passages state the true value and not the new one, and never to a one-sided bound ("more than 500"), either end of a range ("5 million to 10 million"), a year after a word that makes it a bound or a span, even with a month or a day between ("until June 2020", "as of December 31, 2025"), a value said twice, or a claim with three or more quantities. A value after "about" is edited: the new value is at least 25% away, beyond the method's 10% tolerance for "about".
   - `ev_insufficient`: a true claim with passages that do not settle it, of two kinds. *Unrelated* (100 items): the passages saved for another claim, with no source in common, no number in common, and no content word in common across articles (at most one within an article); a record lends its passages once, and this part is cut to 100 items. *Close* (67 items), only for claims that state a number or a year: one passage that shares at least two content words with the claim (scale and approximation words such as "million" or "about" do not count: `close_ignore_words` in `config.json`) and holds none of its values, nor the bare digits of any of its numbers. It is another sentence of the claim's own source page (`near_miss`, 47 items, chosen by `near_miss.py`) or a passage of another record of the same article, from another page (`related`, 20; often about another event in the same story). Among the passages that qualify, one holding a *decoy* comes first: another value of the same kind as the claim's (a year for a year, a percentage for a percentage, a number with a scale word for one, a plain number for a plain one). 30 of the 67 hold one. The builder checks every one of these conditions again on the text a prompt shows.
5. **The rule for labels.** Supported: the evidence shows every part of the claim is true. Refuted: it shows some part of the claim is false. Not enough evidence: anything else, including evidence that shows only part of the claim, and evidence that could only be connected to the claim with something it does not say. Working a number out from what the evidence states is fine.
6. **Order and the gate.** Every choice is the lowest `sha256("evidence-verdicts-v1|" + ...)`, and an item's option order comes from a hash of its own key, so each letter is gold about a third of the time (always answering A, B or C scores 31.9, 33.6 or 34.5). Two hosted models, DeepSeek-V4.1-Flash and glm-5.3-flash on NEAR AI Cloud, read each candidate prompt alone at temperature 0. An item is kept only if both give the gold letter as the `choice` scorer reads it; an answer with no readable letter counts against the item. An edited claim is kept only if neither model calls its unedited claim refuted. Qwen3.8-27B is left out of the gate on purpose: it is a Delta League base model, and a set filtered by it would be fitted to it. Where it disagrees, the item goes to the human reader instead (step 7).
7. **Blind human read.** One person, the author of this proposal, read 30 items in one sitting, with that rule on the screen and no label, family or model answer: 6 of each of the four parts (supported; refuted; not enough evidence with an unrelated passage; with a close one), chosen by hash before the read, and the 6 flagged items (2 that Qwen3.8-27B, a model outside the gate, answered differently, and 4 more that an AI review of this proposal found arguable), all mixed. On the 24 sampled items the reader agreed with the label on 24 (supported 6 of 6; refuted 6 of 6; not enough evidence, unrelated passage 6 of 6; not enough evidence, close passage 6 of 6), and on the flagged items on 4 of 6 (the other 2 are items Qwen3.8-27B had answered differently). In all, 2 disagreements and 0 items the reader could not decide. Every item the reader did not agree on is out of the set, with any edit of a removed claim. Nothing was relabelled, and no other item was read by a person. With 6 items a part, a part where the reader agreed with every label can still have up to 39% of its labels wrong (one-sided 95% exact bound), so the read catches gross errors in a part, not a few. An earlier sitting of 20 items is kept in `handcheck.json` and not used: they all came from one family, so they all had the same answer. Each row keeps the SHA-256 of the prompt it was read on, and counts only while the item's prompt is that one.
8. **Seal.** `evidence-verdicts-v1.sha256` holds the SHA-256 of the item file: `d9d22e1e2ac7cd085916a37a717164e4331e8f2097036cb937356325a3fec6ec`. `.gitattributes` keeps the files here at LF line endings on checkout, so the seal holds on Windows too.

## Counts

| Family | Candidates | Passed the gate | After the cap | Removed by the human read | In the set |
|---|---|---|---|---|---|
| `ev_supported` | 257 | 134 | 134 | 2 | **132** |
| `ev_refuted` | 94 | 79 | 79 | 0 | **79** |
| `ev_insufficient`, unrelated passage | 257 | 248 | 100 | 0 | **100** |
| `ev_insufficient`, close passage | 73 | 67 | 67 | 0 | **67** |

378 items from 194 source sentences of 20 articles, quoting 251 passages from 151 pages on 66 hosts. `ev_supported`: 69 claims about the present state of things and 63 about past events. `ev_refuted`: 5 stale and 2 wrong lines as written, 25 number edits and 47 year edits. `ev_insufficient`: unrelated, 55 with passages from another article and 45 from the same article; close, 47 from the claim's own source page and 20 from another page. Gold letters A / B / C: 46 / 39 / 47, 25 / 26 / 28, 49 / 64 / 54.

Of 681 candidates, the gate dropped 139 where a gate model gave another letter, 12 where one gave no readable letter (glm-5.3-flash: 8 stopped without a boxed letter, 4 ran out of tokens), and 2 edits whose unedited claim a gate model called refuted. `dropped.json` lists each with both models' letters, and `vetoed.json` the 47 records and edits that the rules in steps 3 and 4 stopped before any model saw them.

## Measured

Hosted models, one user message, temperature 0, 4,096 output tokens (`solvers.json`; DeepSeek and Qwen with thinking switched off, GLM as served):

| Model | Role | `ev_supported` | `ev_refuted` | `ev_insufficient` |
|---|---|---|---|---|
| DeepSeek-V4.1-Flash | gate, on all candidates | 139 of 257 | 83 of 94 | 324 of 330 |
| glm-5.3-flash | gate, on all candidates | 157 of 257 | 85 of 94 | 317 of 330 |
| Qwen3.8-27B | not in the gate, on the set | 132 of 132 | 79 of 79 | 167 of 167 |

The gate removes almost half of the supported candidates: a short passage often carries only part of a claim. On the 380 items that passed the gate Qwen3.8-27B gave the gold letter on 378. The other 2 went to the human reader, and both left the set there. On the 378 items of the sealed set it gives the gold letter on every one. A model of that size solves this set; its use is on small models.

Two of Delta League's own Qwen3.5-0.8B versions: the original model (test score 17.4) and the version at the top of the leaderboard on 20 November 2026 (test score 54.7), run with `delta evaluate --validation` on one laptop GPU (Transformers backend), **thinking off**, 1,536 output tokens, the competition's sampling otherwise, one sample per item (all 378 items). Beside them, a rule that uses no model (`python breakdown.py --rule`): if claim and passages share no content word, "not enough evidence"; if a number or year of the claim is missing from the passages and they hold another of the same kind, "refuted"; if it is missing and they hold none of that kind, "not enough evidence"; otherwise "supported".

| | Original model `0ea62f40` | Top of the leaderboard `917d6b4f` | Rule with no model |
|---|---|---|---|
| Items answered | 378 | 378 | 378 |
| Score, mean of the three families (95% interval by group) | 61.7 (57.3 to 66.0) | 73.4 (68.3 to 78.3) | 87.4 (82.9 to 91.4) |
| `ev_supported` right | 105 of 132 | 122 of 132 | 125 of 132 |
| `ev_refuted` right | 30 of 79 | 49 of 79 | 70 of 79 |
| `ev_insufficient` right | 113 of 167 | 110 of 167 | 132 of 167 |
| Rules without evidence (an unfinished answer does not rule) | 35 of 167 (21.0%) | 56 of 167 (33.5%) | 35 of 167 (21.0%) |
| ... on an unrelated passage | 16 of 100 (16.0%) | 32 of 100 (32.0%) | 5 of 100 (5.0%) |
| ... on a close passage | 19 of 67 (28.4%) | 24 of 67 (35.8%) | 30 of 67 (44.8%) |
| Declines a clear case | 29 of 211 (13.7%) | 30 of 211 (14.2%) | 11 of 211 (5.2%) |
| Flips supported and refuted | 15 of 211 (7.1%) | 10 of 211 (4.7%) | 5 of 211 (2.4%) |
| No answer | 51 of 378 (43 cut off at the token limit) | 1 of 378 (1 cut off at the token limit) | none |
| ... on `ev_insufficient` items | 19 of 167 | 1 of 167 | none |
| Output tokens | 278,419 | 110,736 | none |

Item by item, on all 378 items (96 groups of linked items), the original model scores 61.7 and the leader 73.4: +11.7 points (95% interval by group: +4.7 to +19.3). The original model gives no readable letter on 51 items (43 cut off at 1,536 tokens) and the leader on 1, and those count as wrong. Had each model answered its unfinished items at its own accuracy in each family, the scores would be 71.5 and 73.6 (+2.0, taken before rounding). That is kind to the original model, since those items are harder than the rest: the leader gets 65% of them right, against 76% of the others. So had the original model finished, the gap would very likely lie between +2.0 and +11.7 points: much of it is finishing. On `ev_insufficient` items the leader rules without evidence on 56 of 167 and the original model on 35, which leaves 19 of them unfinished (at its own rate it would rule on about 40). Item by item, the leader rules more often in 22 groups and less often in 10 (exact sign test p = 0.050; with one vote per claim instead, p = 0.025). Leaving out the items either model did not finish, it is 19 groups against 10 (p = 0.136; by claim, p = 0.134). Every count leans the same way, but with the stricter of the two units p = 0.050 on all items and p = 0.136 on the items both finished. So the version at the top of the leaderboard may rule more readily on evidence that settles nothing; this set is too small to say more.

The rule with no model scores 87.4, above both models in every family. It reads supported and refuted items by matching values, which is how most of them were made, and gets 95 of the 100 unrelated passages right because they share no words with the claim. On close passages it gets 37 of 67: right on every one without a decoy, wrong on every one with a decoy (30). The simpler rule the close passages were first built against, where any missing value means "refuted", scores 80.1 and gets none of them. A model below the rule's score is not yet using the passages better than matching numbers does.

Limits of these runs: thinking is off because the official profile (thinking on, 8,192 tokens) takes minutes per item on a laptop, so they say nothing about that profile; each item was sampled once at temperature 1.0; and a small model sometimes argues for one option and boxes the letter of another, which the score follows, as the platform's does. The runs were made in parts on 4 item files (earlier builds of this set and files of some of its items, SHA-256 `19ee7e09`, `b00f2353`, `8fb99e27`, `525d3b4a`); `breakdown.py` matches each answer to its item by the SHA-256 of the prompt, so every answer counted is for a prompt exactly as it is in the sealed file, and answers for prompts no longer in it are left out. Items are not independent (some share a claim, some show a passage another item shows), so every interval resamples groups of linked items and every sign test counts one vote per group, and also one vote per claim. The command for the official profile is under "Reproduce". The per-item letters and scores are in `baselines/`, without generated text.

## What this set is and is not

- **Public, so development only.** Do not rank with it.
- **Not a human-annotated set.** Labels come from one AI annotator, receipts that are re-tested mechanically, a gate of two models, and a blind read of 30 items by one person, the proposal's author. The annotator's own audit was run in the same session as its first pass, which the method states as a limit.
- **Selected to be clear.** The gate keeps what two capable models agree on, so hard and ambiguous cases are gone. It measures whether a small model reads carefully, not the limit of evidence reasoning.
- **Narrow.** 20 articles on crypto networks, stablecoins and prediction markets; English only.
- **Much of it yields to matching values, by construction.** Supported items state their claim's values and most refuted items are one-value edits (72 of 79; only 7 are lines that were wrong in the wild), so the rule with no model in "Measured" reads them well. Its score, 87.4, is a floor a model has to clear before its answers show more than value matching. Close passages with a decoy are where matching values fails; `breakdown.py` reports every stratum apart.
- **Close passages are close, not always on the same event.** A `related` passage shares two or more content words with the claim and comes from the same article's sources, so it is often about another event in the same story. A `near_miss` passage is from the claim's own source page.
- **Holding no value of the claim is not the same as implying none.** The check is on numbers as written: a passage can still imply the claimed value by arithmetic (in `ins.tether-0073.near` the figures add up to the claimed $775 million). Such an item stays "not enough evidence" only because another part of the claim, here its date, is missing.
- **Sources are live pages.** A receipt records what a URL returned at one time. `check_receipts.py` asks every source again: on 2026-11-21, of 251 shown passages, the source came back byte for byte for 81, changed but with the receipt's pattern still matching for 158, changed with the pattern gone for 3, as a PDF it cannot read for 7, and not at all for 2. The shown passage itself was still on the page for 222, and not for 17.
- **Recall alone cannot solve it, but it helps.** A model that ignores the passages and answers from memory gets `ev_insufficient` wrong, because those claims are true. On the other two families memory and the passages point the same way, so those two alone cannot tell reading from recall.
- **Passages are cut from saved page text.** Some carry the page's navigation text with the sentence (menu items, link labels such as "Learn More"). The cut follows sentence ends, not the page's layout.
- **Claims are short forms.** They are the annotator's checkable forms of the sentences, and many read like notes ("consensus by proof of work."). The article named in the prompt supplies the subject.
- **People.** Some prompts name people: passages are quoted as their sources wrote them, and some name founders, officials or authors. No claim names a person: the builder refuses a claim, or a close passage, that matches the list of names in `config.json` (`people`), which is a list and not a classifier, and every claim in the set was read for names.
- **Conflict of interest.** The author runs the court these items come from, and Qwen3.8-27B sits on its jury. Any run of ours belongs in an unranked reference row.

## Reproduce

Python 3.12, standard library only, from this folder:

```sh
python build_set.py .                      # rebuilds the six output files from records.json and compares bytes
python -m pytest -q ../../../tests/test_evidence_verdicts.py   # from an environment with deltaleague[local]
python check_receipts.py                   # every shown passage on its source as it is today (network)
```

Run a model and read what it did:

```sh
delta evaluate --workspace WORKSPACE --validation evidence-verdicts-v1.json --no-thinking --max-new-tokens 1536 --output run.json
python breakdown.py run.json               # gold against answer, the error rates by stratum, an interval by group
python breakdown.py run.json --save run.answers.json    # the run as a small file with no generated text
python breakdown.py leader.answers.json --versus base.answers.json
python breakdown.py --rule                 # the rule with no model, as a run
```

Leave out `--no-thinking --max-new-tokens 1536` for the competition's own profile. `delta compare` pairs two reports made on the same file by its hash.

## Extend it, or make a fresh set

`build_set.py` is the builder: candidates, vetoes, the close passages, option order, the gate, the human read and the seal, from `records.json`, `config.json`, `solvers.json` and `handcheck.json`.

```sh
python near_miss.py DIR/records.json --pages PAGES --out DIR/records.json   # fill each record's close passage from saved pages
python build_set.py DIR --candidates candidates.json --no-compare
python ask_models.py candidates.json --endpoint https://cloud-api.near.ai/v1 --api-key-env NEAR_AI_API_KEY \
    --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --extra '{"chat_template_kwargs": {"thinking": false}}' \
    --header "x-no-aliasing: true" --out DIR/solvers.json          # once per gate model in config.json
python hand_check.py --dir DIR --reader NAME                       # the blind read: the same mixed sample as v1's
python build_set.py DIR --out DIR --no-compare
```

- New records are the part that needs research: for each sentence, an annotator (a person or an agent with web access) finds a primary source, writes the short claim and a pattern that matches the source's text, records the verdict following METHOD.md, cuts the passage a prompt shows from the saved page (step 3), and saves the page's text as `PAGES/<receipt key>.txt` for `near_miss.py`. A record's fields are the ones in `records.json`; the builder refuses a record with no receipt it can show, a claim that names a listed person, or a passage from an excluded host.
- `ask_models.py` works against any OpenAI-compatible https endpoint and never follows a redirect, so the key goes only to the endpoint named. v1's runs used the line above for DeepSeek, no `--extra` for `z-ai/glm-5.3-flash`, and `{"chat_template_kwargs": {"enable_thinking": false}}` for `Qwen/Qwen3.8-27B`.
- `hand_check.py` draws the sample the human read used: 6 items of each of the four parts by hash, and every flagged item (`flagged.json`, and any item a hosted model outside the gate answers differently), mixed. A larger `--per` continues the same order. A reader can only remove items.

## Hidden items

Two routes, offered rather than built:

1. **A sealed set from other articles.** The same pipeline on articles outside these 20, never pushed anywhere: the hash here, the items and records delivered out of band, as `benchmarks/heldout-v1` did. It needs fresh annotation, so it exists only if you want it.
2. **Facts that settle after a submission's date.** The court settles new claims on NEAR mainnet, each with three signed verdicts and its evidence readable from the chain: 24 rulings as of 2026-11-21, about 1.6 a day. That is far too few to rank with today, and its labels are the court's own outcomes, so they are circular for any model on the jury. It is a source of items no model has seen, not a ready set.

## Files

| File | What it is |
|---|---|
| `evidence-verdicts-v1.json`, `.sha256` | the items and their seal |
| `records.json` | the 301 checked sentences with their receipts and close passages (the builder's input) |
| `config.json` | every rule's parameter: caps, labels, gate models, excluded hosts, the names vetoed, the words a close passage's shared words leave out |
| `solvers.json` | the letter each hosted model gave to each candidate prompt, keyed by SHA-256 of the prompt |
| `handcheck.json`, `handcheck-sample.json`, `flagged.json` | the blind read (one row per item read, and a set-aside earlier sitting); the sample the reader was shown, with each item's prompt hash, option order and gold letter, which the import compares the reader's letters with (the reading page carries no gold letter and never loads this file); and the items an AI review of this proposal found arguable, with its reasons |
| `provenance.json` | per item: label, option order, source sentence, Wikipedia revision, each passage's URL, hash, fetch time and pattern, and for a close passage its shared words and whether it holds a decoy |
| `dropped.json`, `vetoed.json`, `report.json` | what the gate and the rules removed and why; the counts above |
| `baselines/` | the runs in "Measured", per item, without generated text |
| `build_set.py`, `near_miss.py`, `ask_models.py`, `hand_check.py`, `check_receipts.py`, `breakdown.py` | the builder and its tools, standard library only |
| `.gitattributes`, `NOTICE.md` | LF line endings for the seal; sources, licences and attribution |

Tests: `tests/test_evidence_verdicts.py` rebuilds the set and compares bytes, loads it with the platform loader, scores every item with the platform scorers, runs it through `delta evaluate` with a stand-in model, checks each rule above on every item (and that the builder vetoes a close passage that breaks them), and tests each tool offline.
