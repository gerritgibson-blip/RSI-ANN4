# RSI-ANN4 — TRN: an evolved ternary reservoir agent

TRN is a hardware-efficient recurrent network that learns agentic tasks in a sandbox.

- **Network.** A fixed, quantised (ternary) spiking-style reservoir of about 15k neurons. On top sit linear readouts fitted in closed form by ridge regression; there is no backpropagation.
- **Training.** Evolution mutates the reservoir and its readout genome. A paired, common-random-numbers comparison system decides which mutants survive.
- **Hardware.** Everything runs on a single 24 GB GPU.

Snapshot: 2026-11-27 21:00 UTC, generation 803, curriculum stage 6 of 6.

## Current result

**Record champion (gen 803): fitness 0.9439 on the held-out final set**, across 24 task families plus next-character language modelling (weight 0.1).

| Stage | Families (final-set success) |
|---|---|
| 0 perception | facts 0.995, instr 0.962, toolpick 0.947 |
| 1 tool calling | calc 1.0, lookup 1.0, memory 1.0 |
| 2 multi-step | install 1.0, download 0.993 |
| 3 simulated internet | web 0.993, webinstall 0.953, research 0.913 |
| 4 reasoning / generalisation | hop2 0.993, chain 0.998, toolpick2 0.872, instr2 0.953, webdeep 0.993 |
| 5 unscaffolded extended tasks | multi 1.0, retry 0.96, research2 0.947 |
| 6 general extended agentic tasks | todo 1.0, cond 0.547, recover 1.0, transfer 0.973, inject 1.0 |

Stage 6 tests ordered multi-goal plans, branching on tool output, error recovery, tool-to-tool transfer and prompt-injection robustness.

All environments are procedurally generated and verifiable. Success is judged by environment state. Test episodes use entity names that never appear in training.

Main open weaknesses:
- **cond** (branching on tool output): about chance. The agent has to compare a word it keeps in its note with a word in the tool output.
- **toolpick2** (compositional tool choice): 0.87.
- **research** (multi-step research with memory): 0.91.

### Progress

| Date (UTC) | Gen | Stage | Final fitness | Change |
|---|---|---|---|---|
| 11-24 22:05 | 19 | 3 | 0.720 | word-match fast-memory module (+0.12, z 12.7) |
| 11-24 23:19 | 41 | 3 | 0.761 | query-key attention for argument selection |
| 11-25 01:50 | 86 | 3 | 0.831 | recurrent gain / chain fix |
| 11-25 20:57 | 467 | 3 | 0.940 | bridge (two-hop) binding attention (z 8.9) |
| 11-26 00:37 | 507 | 3 | 0.966 | error-driven training-data allocation |
| 11-26 07:14 | 557 | 4 | 0.958 | occurrence linking + attention mix |
| 11-26 18:59 | 662 | 4 | 0.966 | episode-memory reset, −512 neurons |
| 11-26 23:53 | 692 | 5 | 0.968 | module redraws, bag-of-words, −256 neurons |
| 11-27 07:43 | 740 | 5 | 0.973 | prior-centred contextual readout gate |
| 11-27 20:40 | 803 | 6 | 0.944 | note length 6 (todo 0.47 → 1.0, z 5.5) |

Each new curriculum stage adds harder families, so fitness resets lower at every stage change; it isn't comparable across stages. The full event log is in `state/events.jsonl`.

## Key research findings

The full notes, including negative results, are in [`RESEARCH.md`](RESEARCH.md).

1. **Capabilities beat parameter tuning.** Random edge add/delete has effects below readout-refit noise. The large jumps all came from *capabilities* matched to measured failure modes: a word-match fast memory, query-key attention, bridge attention, occurrence linking, the self-written note, and the contextual gate. Evolution then tunes them.
2. **Measure the failure anatomy first.** Nearly all failures were argument selection, not action choice. Probes that trace failing episodes (`patches/diag_*`, see RESEARCH.md) found the real causes every time. Examples:
   - causal readouts where the deciding word comes later
   - a note that could never be written in time
   - a branch whose comparison word was never stored
3. **The comparison system matters as much as the model.** Common random numbers, a rotating gate set, a separate dev set, racing and a final set with a z ≥ 1.5 advance rule kept selection honest. Several stalls turned out to be measurement bugs, not model limits:
   - epoch-to-epoch allocation swings
   - degenerate training samples (now guarded by `apply_epguard.py`)
   - a start-up re-baseline that stored too few evaluation items, which silently broke paired comparisons (`apply_evalfix.py`)
4. **Data allocation needs inertia.** Error-driven allocation helps, but only with damping: train accuracy taken as the minimum over 4 epochs, and each family's share allowed to fall by at most 15% per epoch.
5. **Some conflicts are capacity limits.** instr and instr2 share a causal answer pointer, and their texts are identical up to the deciding clause. Six attempts each traded one family against the other. That needs a structural change, not more data.
6. **Efficiency before size.** The network got smaller over time (15,872 → 15,104 neurons). Removing neurons and the unused recurrent product sped up generations without losing accuracy.

## Repository layout

- `trn/` — the live, fully patched source.
  - `env.py`: task families
  - `reservoir.py`: ternary reservoir
  - `agent.py`: runner, readouts, attention, note
  - `evolve.py`: evolution and comparison system
  - `mutate.py`: mutation operators
  - `readout.py`: ridge accumulators
- `patches/apply_*.py` — every change made during the run, as an idempotent, documented patch, in order. Each file's docstring records the evidence for that change.
- `RESEARCH.md` — chronological research notes and results.
- `state/state.json`, `state/events.jsonl` — evolution state and the complete event log.
- `models/champion_record_gen803.pt.zst` — the record champion (zstd-compressed `torch.save`, about 290 MB unpacked).
- `models/champion_current.pt.zst` — the live champion, refitted on epoch 81's training sample.
- `watch_advance.sh` — helper that exits when a new ADVANCE is logged.

## Using the model

```bash
zstd -d models/champion_record_gen803.pt.zst -o champion.pt
sha256sum -c models/SHA256SUMS     # checks the .zst files
```

```python
import sys; sys.path.insert(0, ".")
from trn import evolve as E
res, heads = E.load_net("champion.pt")
E.EVAL_MULT = {"facts": 4, "instr": 4, "toolpick": 4, "chain": 4, "toolpick2": 4, "instr2": 4}
print(E.evaluate(res, heads, 6, "final")["families"])
```

To resume evolution:
1. Put `champion.pt`, `champion_record.pt` and `state/state.json` into a state directory.
2. Run `TRN_STATE=<dir> python -m trn.evolve` (CUDA GPU needed). The agentic text corpus is regenerated deterministically on first start.

## Safety note

All tools (package install, web, files, memory) are simulated inside Python; nothing touches a real system or network. The agent has no access to money, accounts or real compute, and acquiring any of them is deliberately out of scope.
