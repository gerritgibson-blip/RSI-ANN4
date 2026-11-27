# TRN — research notes: raising the rate of improvement and agentic competence

Written 2026-11-24 while evolution runs (generation ~40, stage 3, N = 13,312 + word-match module).

## 1. Evidence so far

| Finding | Source |
|---|---|
| Random edge add/delete (0.5–15% of synapses) has effects below readout-retraining noise (sd ≈ 0.008 fitness) | `diag_mutation.py` (14 networks) |
| Under the v3 comparison system, edge ops are consistently negative (z = −2.7, −1.7) | gens 23, 32 |
| One **structural capability** (the word-match fast memory) gave +0.12 fitness (z = 12) | gen 19 |
| Rule / subpopulation / module-parameter changes: effects of ±0.005–0.016, rarely confirmed on the held-out set | gens 20–40 |
| DAgger rounds never improved the stage-3 readout over oracle-only fitting | plateau logs |
| **100% of remaining failures are argument selection**; the action choice is always right | failure traces (lookup, instr, toolpick, facts, web) |

The failure traces show *why* argument selection fails. The per-word pointer scores each word as it streams past, but the disambiguating information often comes later, or has to be compared with something seen elsewhere:

- *lookup*: "muzil : capital talhil ; population 24 million … question : population ." The key arrives after the values.
- *web*: "rika costs 316 ; … find rika ." The item is named after the list.
- *toolpick*: the request type has to be combined with each option's identity. A linear readout can't express that conjunction.
- *instr*: deciding which of two familiar names the "not" / "no wait" applies to.

**Conclusion.** The largest gains come from adding *capabilities* matched to measured failure modes. Fine-tuning parameters gives small, hard-to-confirm gains, and random wiring gives nothing. The evolutionary search should be steered toward capability-level variation, and the measured failure modes should decide which capabilities to offer it.

## 2. Implemented now (being tested by the evolution itself)

1. **Query-key attention for argument selection** (from transformer attention / pointer networks).
   - Every word gets a key: a fixed random ternary projection of the reservoir features at the word's end, concatenated with the projection at its clause end, centred over the observation.
   - At decision time a ridge-trained **query head** maps the decision state to a query (its target is the mean key of the correct argument words). Words are ranked by query·key, combined with the old pointer score.
   - It is still a pure readout: nothing feeds back into the reservoir. It costs no extra Gram matrix, because the query head shares the decision Gram.
   - Small-network test (N = 2048, 100 tasks/family):

     | Task | Pointer only | Pointer + attention |
     |---|---|---|
     | Lookup | 45% | 78% |
     | Download | 57% | 77% |
     | Instructions | 50% | 64% |
     | Web | 34% | 40% |
     | Calculator | 99% | 85% |
     | Install | 95% | 83% |

     The multi-word copy actions get worse, so attention is applied **per action**.
   - The per-action weights are a **readout genome** (`rules["attn_mix"]`) mutated by the `readout` operator. The network is untouched and not retuned, so the comparison against the champion is a clean A/B on identical training data. First test: attention on `answer` at weight 1.
2. **Directed operator selection.** Each mutation operator's probability is scaled by exp(mean paired z of its last 12 uses), shrunk toward its prior. Operators that keep failing, like random edges, fade; productive ones, like module and readout, are tried more.
3. **Module context from delay lines.** The word-match module can take its context snapshot from the delay-line neurons, which hold ordered recent history, instead of random mixing neurons. This targets *instr*, where the relevant context is "not" two words earlier.
4. **Racing (successive halving), live since 2026-11-25 01:47.**
   - Each generation spends about 2.5 of its 2.7 minutes collecting readout training data; the gate test is cheap.
   - Training data is now collected in two chunks (1/3, then 2/3). The readout Gram statistics are additive, so after chunk 1 the mutant is solved and gate-tested against the champion fitted on the same chunk 1 (common random numbers).
   - If the mutant's paired z < −2.0, it is dropped and the remaining 2/3 of collection is skipped. Survivors continue at almost no extra cost (one extra solve + gate pass ≈ 0.2 min).
   - In generations 54–82, 11 of 29 mutants finished at z ≤ −2. If chunk-1 z tracks the final z, about 40% of generations finish in about 1.1 minutes instead of 2.7, roughly 1.3× more mutants per hour.
   - Every generation logs `z1` (chunk-1 z) next to `z` for calibration. If z1 predicts z well, the threshold can be raised toward −1.
5. **Readout mutations are never no-ops.** Generations 80–81 re-tested an identical network, because the new weight equalled the current one.

## 3. Next candidates, ranked by expected value / cost

| # | Idea | Targets | Cost | Notes |
|---|---|---|---|---|
| 2 | **Thought / scratchpad loop** (chain-of-thought analogue): a "thought" readout emits a word (e.g. the question key, the target item) that is fed back as input before acting | late information, multi-hop | medium | a direct use of the spec's "thoughts" readouts; oracle thought labels = argument words |
| 3 | **Surrogate-guided tuning of continuous knobs** (Bayesian optimisation / CMA-ES over leak-shift distribution, trace decay, module decay, echo target, gains) instead of ±1 random steps | passive-decay tuning | low–medium | see §4 |
| 4 | **Bilinear / conjunctive readout features**: quadratic features of a low-rank learned projection (an MLP-like readout layer, still closed-form) | toolpick-type conjunctions | medium | Gram size grows with rank² |
| 5 | **Word-level input embeddings**: inject the hashed word signature as an input pattern at each word end | identity of novel words | low | like token embeddings for unseen names |
| 6 | **One-Gram-at-a-time readout solve** (others parked in system RAM) | memory | medium | frees about 6 GB: concurrent experiments, later growth to about 20k neurons |
| 7 | **Readout bagging** across epochs (average heads fitted on different samples) | readout noise | low | cheaper than a larger training set |
| 8 | Population-based evolution with crossover of rule/module genes | search breadth | medium | only after 1–3 |

## 4. Passive decay (leak) — what the data says and what to try

- Leak today:
  - Each mixing neuron decays by |v| >> k with k uniform in 1–8, i.e. time constants of 2–256 ticks.
  - Delay-line neurons use k = 1.
  - Readout traces decay with shift 3, about 8 ticks.
  - The word-match memory decays by 2^-6 per word, about 64 words.
  - The echo target (≈ 200 ticks) sets the global gain.
- Mutations of single scalars (echo target, trace shift, think ticks, chain length) moved fitness by ≤ 0.007. The one accepted chain-length change was not confirmed on the held-out set. Global decay settings are near a broad optimum; the remaining gains are in *where* long memory is available, not its overall rate.
- More promising:
  - (a) **Evolve the leak distribution per sub-population**, e.g. a dedicated slow population (k = 8–12) wired from the word-match outputs. The `subpop` operator already draws new leak ranges, so this extends it with a "slow memory" preset.
  - (b) **Multi-timescale readout traces**: expose a second, slow trace for a subset of neurons. The readout then sees fast and slow history at no extra network cost; the feature count is held constant by swapping out part of the membrane features.
  - (c) **Activity-dependent (adaptive) decay**: neurons that just fired leak slower, giving a cheap working-memory latch. This is an integer rule (k += 1 while the trace is high), so still hardware-efficient.

### 4.1 Result: recurrent gain was far too high (gen 86, 2026-11-25)
- In gen 86, chain_max went 128 → 226. A 226-tick delay line alone outlasts the 200-tick echo target, so the tuner pushed g_rec to its floor: 23 → 1.
- Fitness went 0.797 → 0.831 on the final set (z = 4.0). Next-character prediction went 64% → 81%, and web went 67% → 77%.
- Reading: strong recurrence blurred the exact history that the delay lines carry. The near-feed-forward regime keeps that history for the readout, much like an n-gram / time-delay network with nonlinear mixing.
- The echo-time criterion was the wrong objective. g_rec is now a gene (`rules["g_fix"]` ∈ {None, 0, 1, 2, 4, 8, 16}; None = echo-target search).
- Follow-ups:
  - Longer and more delay lines; the hardware cost is just neurons.
  - Multi-tap readout over delay lines.
  - Treating the leak k of mixing neurons as the main memory knob, now that recurrence no longer sustains activity.

### 4.2 Firing-rate optimum moves with architecture (gens 101 → 295)
- Target firing rate went 6% → 4.2% (helped, gen 101), then back up to 5.9% → 8.2% (helped, gens 250 and 295) once the network had become delay-line dominated.
- Fitness reached 0.870. Instructions went 60% → 72%, tool choice 48% → 55%, web 88% → 94%.

### 4.3 Failure anatomy at N = 15,872 (gen 461, 400 fresh tasks per family)
- **Facts** (49.5%): 95% of errors answer a value of the *wrong entity*. Success is 11–29% when the answer sits in the first fact versus 73–84% in the last, so the network mostly answers with the most recent value. Reverse questions ("which package needs V") are worst at 26–39%, usually answering V itself.
- **Tool choice** (47%): success depends on the answer's position in the option list: 23% first, 42%, 61%, 78%, then 29% last. The choice is driven by position, not by the request class.
- **Instructions** (63.5%): "x but not y" is weakest at 39–47%; "no wait" is 76%.
- Diagnosis: the capability missing is **binding**, i.e. relating a word to another occurrence of the same entity, or to a class. Perception and action choice are fine (decision validation accuracy is 0.9998).

### 4.4 Bridge (two-hop) binding attention: in test
- A new pointer channel, `focus`, is trained to point at the word that links the question to the answer's clause, for example the package name in "what is the version of X". Its labels come automatically from the oracle answer.
- That word's hashed identity signature becomes a query, matched against each candidate's clause-context signature. Identity matching generalises exactly to names never seen in training.
- `key_id` adds each word's own identity signature to the attention keys, so the query head can learn request-class → option-word mappings.
- Both are readout genes (`bind_mix`, `key_id`) under a new `bind` operator.
- Tiny-network test with the true bridge word (oracle diagnostic): web 38% → 81%, facts 31% → 50% (z = 3.1). The learned focus pointer finds the bridge word 100% of the time on lookup, web, download and memory, but only 13% of the time on facts at N = 1024.
- **Champion A/B (N = 15,872, same training data, 150 fresh tasks per family, 2026-11-25):**

  | Arm | Fitness | Facts | Tool choice | Lookup | Web | Instr |
  |---|---|---|---|---|---|---|
  | base | 0.865 | 46% | 49% | 99% | 92% | 73% |
  | bind (answer, lookup, open) | 0.850 | **93%** | 47% | 79% | 77% | 62% |
  | key_id | 0.883 (z 2.8) | 45% | **92%** | 93% | 85% | 65% |
  | both | 0.889 (z 2.7) | **97%** | **92%** | 75% | 80% | 61% |

  - Each mechanism roughly doubles its target family, the largest single-step gains since the word-match module.
  - Both hurt actions whose arguments are novel words with no bridge (lookup, open), so they are now **per action**: `bind_mix` per action, and `key_mix` weights the identity part of the attention score per action (default answer only).
  - Deployed live with forced trial `bind:answer=2` + `bind:key_id=1`.
  - Further refinement is left to the `bind` operator.
- Implementation notes:
  - Ridge `fit(groups=...)` picks a separate regularisation for the focus channel; a shared alpha degraded the argument pointers.
  - Focus confidence scales the binding term (pointer outputs are about 0.1–0.2, because labels are sparse).

### 4.5 Learned contextual gate over selection components (2026-11-25)
- `ptr_all` (every occurrence of a single answer word is a pointer target): neutral on the champion (instr 59 → 61%, z = 0.45). Kept as a gene, default off.
- Stage-0 lab (N = 4096, facts/instr/toolpick only, `scratchpad/lab0.py`) recorded which component alone picks the right name on instr:

  | Component | "do not" | "no wait" | "but not" |
  |---|---|---|---|
  | unary pointer | 96% | 95% | 64% |
  | attention | 81% | 75% | 61% |
  | identity key | 51% | 53% | 34% |

  The fixed per-action sum drowned the good unary signal, giving a family score of 40%. Facts, toolpick and instr all use `answer`, so per-action weights cannot separate them.
- **Gate** (`apply_gate.py`, gene `rules["gate"]`):
  - score(w) = Σ_c Z_c(w)·g_c(s), with g linear in [R-projection of the decision state, action one-hot, 1]. This is mixture-of-experts routing, about 550 weights.
  - Fitted in closed form on a short pass of fresh oracle episodes (EP/8 per family) after the heads.
  - Lab: instr 40 → 79%, toolpick 75 → 85%, facts 88 → 90%; fitness 0.66 → 0.82.
  - Live trial (gen 502) **rejected**, z = −7.8 (facts 98 → 57%). The lab was unrepresentative: its stage-0-only training gave instr a third of the data.
  - On the champion's own heads (`scratchpad/gate_champ.py`), gate v2 (per-decision centred ranking rows, uniform family weights) still lowered facts (94 → 75%). Kept as a gene, default off.
- **Real cause of the instr failure: data starvation.** On the champion the unary pointer alone gets only 43–53% on instr, versus 95% in the stage-0 lab. The fixed curriculum emphasis gave instr 3% of training episodes, while stage-3 families (already at 100%) got 18% each.

### 4.6 Error-driven training allocation (`apply_alloc.py`, 2026-11-26)
- Episodes per family ∝ max(1, emph/3) + 12·(1 − champion gate accuracy). Recomputed at each epoch start and fixed within the epoch, so comparisons stay paired.
- Champion A/B (same network): instr 59 → **79%**, toolpick 93 → **100%**, facts unchanged, no family lost; fitness 0.937 → 0.958 (z = 4.7).
- General lesson: curriculum emphasis must follow measured errors, not stage order. The same principle as hard-example mining / prioritised replay.

## 4b. Stage 4 (2026-11-26): harder reasoning + generalisation
- Stage 3 closed at final fitness **0.966**. Snapshot: `trn_state/archive_v1/champion_stage3_final.pt` and `state_stage3_final.json`.
- New families (`apply_stage4.py`); all earlier families stay active:

  | Family | Tests |
  |---|---|
  | `hop2` | two-hop binding; one-hop lands on the wrong word |
  | `chain` | goal carried across observations |
  | `toolpick2` | compositional generalisation; test (frame, phrase) pairs never seen in training |
  | `instr2` | double / novel corrections, three names |
  | `webdeep` | navigation up to 3 links deep, decoy prices |

- Oracles verified: 400/400 per split, all arguments visible.
- Expected capability gaps: two-hop binding (bind is one-hop → a `hops` gene), and goal memory across observations for `chain`.
- Stage-4 baseline (final set): 0.879.

### 4b.1 Multi-hop binding (`apply_reason.py`, gene `hops = 2`)
- Bridge labels come from a BFS over clauses linked by rare words (count 2).
- The focus pointer is trained on the chain start; a 3-way `hop` decision head predicts the depth (0/1/2), and the query is propagated h times.
- Champion A/B: hop2 75 → **99%**, facts 77 → 91%, instr 69 → 85% (depth 0 switches binding off where it was noise); z = 8.4.
- **Live gen 519 ADVANCE: final 0.879 → 0.924 (z 7.7).**

### 4b.2 Goal memory as a self-written note (`apply_reason` + `apply_carry2` + `apply_carry3`, gene `carry = 1`)
- A carry pointer channel writes words into a note prepended to later observations.
- Labels come from an oracle roll-out on a copy of the environment: carry a word that a future answer clause needs and that the future observation lacks.
- v1 (note must be re-copied every step, fixed threshold): chain unchanged, z −2.8. The diagnostic showed step-1 notes 100% right but re-copying ~50%, and spurious notes on 64% of web decisions.
- v2 (persistent note of ≤ 4 words, a 3-way `ncarry` head decides how many words to write): chain 66 → 83%, but webdeep 99 → 61%. A noted item name occurred 3×, so it no longer counted as a rare link word.
- v3 (rare-word counting ignores the note prefix; a noted word counts as linked if it occurs once in the text): chain 66 → **83%**, webinstall 93 → 99%, instr2 59 → 65%, no losses; z = 3.7. Deployed with forced trial `bind:carry=1`.
- Live: accepted at gen 527 on the dev override. Gen 530 (reroute 21 delay links) then **ADVANCED: final 0.924 → 0.936 (z 2.5)**, with chain 85% and toolpick2 93%.

### 4b.3 Occurrence linking (`apply_occ.py`, gene `occ`)
- Failure anatomy (gen 530): every instr2 failure copies the wrong *option*. The pointer was trained on the last occurrence (the option list), where all names share the same context. "v x , not y and not z" scored 26%, below chance.
- occ = 1: every occurrence of a single-word answer is a target, and at inference a word's unary score is the max over its occurrences.
  - Champion A/B: fitness +0.0056, z = 2.7, instr2 58 → 64%.
  - occ = 2 (also pooling the attention score): z = 0.8. ptr_all alone: z = 1.7.
- Live: accepted at gen 547 (gate z = 1.4, dev z = 0.8); final 0.932, no advance.
- Probe (gen 550): the attention term mostly prefers the last option. The instruction-occurrence unary score is often right, but option-list occurrences (trained positive under occ = 1) score 0.2–0.4 for every name and win the max.
  → occ = 3 (`apply_occ3.py`): the last occurrence of a repeated answer is *not* a target, so the option list becomes a negative for every name.
- occ = 3 v1 also changed `open` targets (not pooled): web 99 → 67%, webdeep 100 → 53%. Fixed by restricting the exclusion to the answer action.
- occ = 3 champion A/B: fitness 0.934 → 0.948, z = 5.2, instr 75 → 86%, instr2 69 → 85%. With answer attention weight 1 as well: instr 92%, instr2 95%, but toolpick2 91 → 83% (z = 5.1).
- **Live gen 555 ADVANCE: final 0.936 → 0.950 (z 3.3; gate z 4.6, dev z 4.9).** Final set: instr2 86%, chain 95%, instr 85%, toolpick2 87%.

### 4b.4 Bridge words must repeat (`apply_bmask.py`, gene `bmask`)
- Probe: chain "river" questions succeeded 51%, against 100% for mayor and 88% for population.
  - The note always held the key and the hop head said 1, but the focus pointer chose "capital", the adjacent note word, which occurs nowhere else. So the identity query bound the note's own clause.
  - Mayor and population answers survive on unary cues (person names, "million"); made-up river names depend on binding.
- Fix: the focus distribution is restricted to words occurring ≥ 2 times in the observation.
- Champion A/B: chain 81 → **92%**, every other family identical, z = 3.6.
- Live: accepted at gen 550 (gate z = 3.5, dev z = 2.8). Final 0.941 against the record 0.936, final z 1.26 (just below the advance bar of 1.5).

### 4b.5 Stall after gen 557: measurement resolution and downward drift (2026-11-26, `apply_eval2.py`)
- Gens 557–634 (7 h): 78 decisions, 4 accepted, no ADVANCE.
  - Structural operators (subpop, module, edges, leak) averaged z ≈ −1, at ~6 min each. Readout-only trials cost ~0.5 min.
- Cause 1, resolution: the remaining ~4% error sits in 6 families. The remaining gains (~0.003–0.005) are below the noise of 150 tasks per family.
  - The champion's own instr score moved between 0.79 and 0.97 from one gate set to the next.
- Cause 2, drift: ACCEPT_Z 0.5 on that noise let harmful mutants in (gen 614: final z −2.05 at acceptance).
  - Re-measured on identical tasks, the drifted champion was *worse* than the record (0.950 vs 0.953, z −1.6).
- Cause 3, winner's curse: re-measured on a 4× larger final set, the gen-557 record is 0.953, not 0.958.
- Fixes:
  - Stratified evaluation: families with record accuracy < 0.99 get 4× tasks (5,164 final items), and each family still weighs equally.
  - The record network was restored as champion (drifted network archived).
  - ACCEPT_Z raised 0.5 → 1.0.

### 4b.6 Allocation oscillation and the episode-memory leak (2026-11-26)
- After the re-measurement, a probe solved instr at ~95% on fresh tasks while the gate said 79%.
  - Same-seed and batch-mix controls ruled out seeds and batch composition (instr 0.958 / 0.962 / 0.963).
  - The swing is between *epochs*: the champion's readouts are refit on each epoch's training sample.
- Cause: bang-bang allocation feedback. Allocation used last epoch's accuracy only, so instr went 0.82 → 0.97 → 0.77 → 0.99 → 0.80 → 0.97 → 0.79 → 0.97, and instr2 traded against it.
  - In every other epoch the champion was ~0.01 worse, so a mutant accepted in a bad epoch could not beat a record measured in a good one.
  - This handicap is larger than any remaining gain, and was the main reason for the stall.
- Fix (`apply_alloc2.py`): allocation uses the *minimum* accuracy over the last 4 epochs. Re-baselined champion gate: 0.9545 → 0.9585 (instr 0.99, instr2 0.94).
- Leak (`apply_sepreset.py`, gene `sep_reset`): an episode starts with [SEP, PAD, PAD]. The word-match fast-weight memory M (decay 2^-5 per word) and the bag-of-words trace were never cleared, so common words retrieved associations from the previous task.
  - Clearing at SEP, without retraining: +0.003.
  - Champion A/B with retraining: 0.957 → **0.963 (z 3.6)**; instr2 90 → 95%, toolpick2 87 → 90%, instr 96 → 99%, chain 91 → 93%.
- toolpick2 anatomy: seen (frame, phrase) combinations 96%, unseen 91%. The weak spot is frame "{x} , thanks ." (phrase first, then filler), where memory requests in unseen combinations score 20%.

## 4c. Stage 5 (2026-11-26): extended agentic tasks without scaffolding (`apply_stage5.py`)
- New families; tool replies are bare, with no restated task and no next-step hints:
  - multi: install a tool, *then* open file F on site S and report a field.
  - retry: flaky lookups time out up to twice.
  - research2: the 7-step research plan with no hints.
- Mechanism `carry = 2`: the note may carry a future action's *argument* that will be out of view.
  - Oracle roll-out up to 8 steps. An argument is carried only at the last step it is visible; bridge words only for answer steps; up to 3 words per step (4-way `ncarry` head).
  - CPU check: multi and research2 are unsolvable with bridge-only notes and always solvable with argument notes.
- Stage-5 A/B (gate set):
  - Champion with carry 1: multi 0%, research2 0%, retry 99%.
  - carry 2: multi 15%, research2 39%, but research 100 → 17%.
- `dedup` (`apply_dedup.py`): the research collapse was two-argument copies picking one name twice ("recall dunzo dunzo"), since a name occurs in both note and text. Punctuation was also being copied.
  - Distinct non-punctuation words: research 17 → 93% (z 8.3); stage 4 unchanged.
- `carry = 3` (`apply_carry4.py`): the multi note held the right words but in pointer-score order, so `open` copied the field word.
  - Writing the note in text order: **multi 12 → 100%, research2 40 → 99%**, fitness 0.870 → 0.937 (z 13.5).
  - Open issues: research 88 → 79%; toolpick2 ~63% (diluted by the new families' share of training data).
- Live gen 662 (before stage 5): ADVANCE 0.963 → 0.966 (z 1.8) from pruning 512 neurons (N 15,360).

## 5. Negative results worth keeping

- **Bag-of-recent-words readout features** (pooled hashed word signatures, gene `mod["bow"]`):
  - Tested at 8 words (z = −0.8, and −1.6 combined with another change) and 32 words (z = −0.05); tool choice gained only about 1 point.
  - So tool choice is not limited by decoding the request; the limit is the request-class × option-word conjunction at pointing time.

- DAgger refinement at stage 3.
- Random edge mutation at any size.
- Rewiring the least-used neurons (by readout weight).
- Pure attention without the per-word pointer: it collapses on multi-word copying.

### 4c.1 Stall after gen 666: allocation swing and a dead operator (2026-11-26 21:30)
- Epoch 66 cut research2's training share from 16% to 7.5% because it read 0.99. Every mutant and the re-fitted
  champion then fell to research2 ~0.55 (research 0.87 -> 0.99 and toolpick2 0.76 -> 0.87 took the data). Epoch 67
  restores 17.5%. Fix (apply_alloc3.py, ALLOC3): a family's share may fall at most 40% per epoch.
- The evolved g_fix = 0 (since ~gen 480) zeroes the recurrent drive, so the reservoir is a bank of leaky integrators
  plus delay chains plus module. The 'edges' operator only changes W and was a no-op: gens 670/671/675 were
  bit-identical (0.9467, z -1.11). It is no longer drawn while g_rec = 0, and the recurrent product is skipped
  (exact; eval 9.5 -> 9.1 s, since CSR was already cheap).

### 4c.2 Allocation needs inertia (2026-11-27 02:00)
- ALLOC3 bootstrapped without previous shares (epoch 68 computed from scratch: instr 9% -> 2.7%, instr 0.99 -> 0.79);
  epoch 68 was ended early. Epoch 70 then cut research 6.3 -> 3.8% and toolpick2 9.9 -> 7.8% (within the 40% floor).
- A/B on the champion (seed 777, same gate tasks): epoch-70 shares 0.9462; floor 0.85*previous 0.9634 (z +7.8,
  research 0.76 -> 0.99); 1.5x episodes 0.9645 (z +8.2, +46% time). Families sit at a data threshold, so the
  share floor is now 0.85 (apply_alloc4.py) and epoch 70 was restarted from epoch-69 shares.

### 4c.3 Gate v3: prior-centred contextual gate with hard negatives (2026-11-27 07:30)
- Probes on the bow-7 champion: toolpick2 frame '{x} , thanks .' fails even on the TRAIN split (browser 67%, test
  memory 37%); instr2 ', not' form 84%. In 18/25 instr2 failures the unary pointer was right and attention overrode
  it; on toolpick2 more attention helps (gen 735: answer attention x2 -> toolpick2 +4, instr2 -5). Both use the
  'answer' action, so the component weighting must depend on context.
- The gate v1/v2 failures (facts 98 -> 57%) had two causes: ridge shrinkage toward ZERO weights, and no candidate
  mask or occurrence pooling. v3 (gene gate=2, apply_gate2.py) shrinks toward a scaled copy of the default mix and
  masks/pools like the default scorer.
- A/B on the champion (one readout fit, same gate tasks): gate off 0.9700. Prior only 0.9715 (z +2.3). v3 fitted on all
  candidates: toolpick2 0.98 but facts 0.84 (z -5): squared error over every candidate is dominated by easy negatives
  in long passages. v3 fitted on the answer plus the 4 hardest negatives under the default mix: 0.9742 (alpha 1,
  z +3.6) / 0.9748 (alpha 0.1, z +4.1); toolpick2 0.91 -> 0.96-0.97, instr2 0.94 -> 0.96-0.97, toolpick 0.96 ->
  0.99, facts 0.99 -> 0.987. Deployed with alpha 1, 4 hard negatives (forced live trial bind:gate).
- Also (4c.2b): refitting the record on new training data reproduced its final set bit-identically (z 0.00), so
  the record is not inflated by training-sample luck; the data SHARES matter, the sample does not.

### 4c.4 instr vs instr2 is a capacity conflict, not starvation (2026-11-27 11:30)
- Gate data size: 1x / 2x / 4x of EP/8 per family -> 0.9731 / 0.9734 / 0.9734 (all z ~ +7.3 vs gate off): not limiting.
- Gate off, the champion already has instr 0.925; the gate lifts it to ~0.95. instr's share had decayed 9.0 -> 4.5%
  while it was solved. Restoring 8.6%: instr 0.943 -> >0.995 but instr2 0.985 -> 0.933 (net z -0.74). The two
  families compete for the same readout capacity; the data share only picks the winner. Not deployed.
- Bug fixed (apply_gate3.py): readout-only mutations reused the champion's heads although the gate head was fitted
  for the old attention mix (gen 755: z -10.8). They now refit while the gate is on. Gate episode budget is now a gene (gate_ep).

### 4c.5 Negative results on the instr / instr2 conflict (2026-11-27 12:00-13:50)
- occ_rec (recency-aware occurrence pooling), on the champion's existing heads: beta 1 z +2.4. In a live trial with
  retrained readouts (gen 764): z -0.11 (instr +0.7, instr2 -0.8). The v1 design gave the option-list occurrence
  (end of prompt) the largest bonus; v2 (apply_occrec2.py, bonus only on non-final occurrences of repeated words)
  gave z -1.6 with full fits. Not deployed (v1 stays in the gene pool, neutral).
- aff (affirmed-mention labels, apply_aff.py; the latest affirmed mention = answer on 400/400 samples): full fits
  instr 0.98 -> 0.84, instr2 0.96 -> 0.84 (z -10), identical with v1 and v2 recency pooling. Multi-positive answer
  labels on these families probably break the attention / bind / gate training, which assumes one target. Not deployed.
- Conclusion: the remaining instr/instr2 gap (~0.02-0.04 each) is a capacity conflict in the shared causal answer
  pointer. Weight, data-share, gate-data and recency fixes all trade one family against the other.

### 4c.6 Degenerate training sample (epoch 78) and the epoch guard (2026-11-27)
Epoch 78 (train_seed 617783): the champion refit dropped to gate 0.958 (instr 0.80, instr2 0.90) and every mutant of the epoch looked the same.
Diagnostic (ep78.py): the gate set was not the cause. Record gen 740 with its stored heads scored instr 0.995 / instr2 0.963 on the epoch-78 gate set. The same genome refit on the epoch-78 sample scored 0.805 / 0.862, gate on or off.
So some rare training samples break the answer readout for instr/instr2. Fix: apply_epguard.py (EPGUARD_PATCH). If the epoch champion's gate fitness is more than 0.01 below the record's final fitness, redraw train_seed (2 retries, keep the best). Degenerate epochs are kept out of acc_hist.
Deployed at 16:05 with a forced rotation to epoch 79 (bad_epoch=True). Epoch 79 refit was normal. The first mutant, gen 784 (module ctx -> mixing), was accepted: gate z +2.13, dev z +2.15, final 0.9728 (z +0.13 vs record, not an ADVANCE).
Note: gen 779 (subpop) was accepted inside the degenerate epoch. It has since been superseded by gen 784.

### 5. Stage 6: general extended agentic tasks (2026-11-27)
The user asked for more extensive general agentic tasks. apply_stage6.py adds five families as stage 6. All use the existing action set, and all are checked at 100% oracle success on CPU.
- todo: ordered multi-goal plan from one sentence (install / save a code / report a file field). Later subgoals are not restated, and answering early fails.
- cond: branch on tool output (language match -> save the capital under a key, else report the population).
- recover: the matching package is deprecated, and the error names the replacement.
- transfer: open a file, then save one of its fields under a key.
- inject: the opened file contains an injected instruction (install X / answer a decoy). Following it fails.
Deployed 17:19 with old-family shares x0.7 and 0.06 per new family. The stage change was automatic ("competent").
apply_stage6b.py fixes a bug: todo's save step required [key, value], but the agent emits text order. Re-baselined at 17:48.
Stage-6 record (final): 0.927. todo 0.473, cond 0.48, recover 1.0, transfer 0.987, inject 1.0. Old families all >= 0.95 except toolpick2 0.867, instr2 0.913, instr 0.967.
Open: cond needs a comparison between a word in the tool output and a word in the task. todo needs long carries (the note holds at most 4 words).

### 5.1 Stage-6 failure anatomy, memo/note fixes, and an evaluation-alignment bug (2026-11-27)
Probe diag_s6.py ran 300 test episodes per family on the champion.
- cond: match 0.19 / nomatch 0.77. At the branch step the task's language word is out of view and was never written into the note. The note was trained only on future arguments and bridge words.
- todo: 2-goal plans 1.0, 3-goal plans 0.0. Every failure is at "open the file F on S".
Fixes:
- apply_memo.py (deployed): an env may define memo(), words a future decision depends on, and they are carried like arguments. Added gene note_len (4/6/8). Result: cond 0.48 -> 0.57-0.65.
- apply_memo2.py (deployed): the per-step note write limit was 3 (2 + carry2), so F and S were never written. The limit is now note_len - 1 when note_len > 4. On a CPU replay of the training labels, 3-goal todos with every argument in view went from 0/204 to 204/204.
  - The first deploy missed the dec Ridge width and crash-looped for about 2 min. Fixed.
- apply_evalfix.py: main() loaded EVAL_MULT only after the start-up re-baselines. The best=None re-baselines at 17:48 and 19:02 stored 3664 items, against 6364 for every later evaluation. paired_z zips the two lists, so all comparisons from 17:48 to 20:05 were mispaired.
  - Gen 800 (note_len 6) logged delta -0.0024. Re-measured: +0.0173, z +6.28 (todo 0.46 -> 0.99).
  - Deployed with a fresh re-baseline and a forced bind:note_len=6.
Lesson: any re-baseline must happen with the full evaluation protocol loaded. Check len(items) whenever a delta disagrees with the per-family numbers.
