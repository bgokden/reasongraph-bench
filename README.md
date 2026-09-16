# reasongraph-bench

Measured behaviour of [ReasonGraph](https://github.com/bgokden/reasongraph), the graph
memory for AI agents, on multi-source *why*-questions in six languages: the case sets,
the evaluation scripts, one results file per experiment, and the experiments that did
**not** help, recorded next to the ones that did.

Maintained by [PrimAxiom Labs](https://primaxiom.ai). The hosted service is
[ReasonGraph Cloud](https://memory.primaxiom.ai).

## The task

A tenant holds a few hundred facts written independently (meeting notes, tickets, an
agent's own notes). A question asks *why* something happened. The answer is a chain of
facts that crosses sources through shared entities and cause-effect links. The metric
that matters is **cause@1**: is the fact ranked first the gold cause. Everything else
(seed quality, walk depth, context size) is instrumentation for that number.

The standing pool is 341 cases; the row quoted below is the **real-swap 335** (the
synthetic French cell replaced by real French prose, which is harder). Case sets, their
origin and licences are in [`cases/`](cases/README.md).

## Leaderboard: embedder, cause@1 on real-swap 335

All rows use the same extraction, reranker and walk; only the embedder and its own
re-fitted thresholds change. Small models only, all run on CPU in production.
These rows were measured with the internal v3 causal extractor (see *Reproducing*);
the rows anyone can reproduce today with the public v2 extractor are in the next table.

| embedder | dim | ALL | en | de | es | fr | nl | tr | source |
|---|---|---|---|---|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` (production until Sep 2026) | 384 | 33.0% | | | | | | | R44 |
| `multilingual-e5-small`, constants re-fitted | 384 | 37.9% | 45 | 33 | 44 | 15 | 45 | 38 | R42 |
| `multilingual-e5-large`, constants re-fitted | 1024 | 42.4% | 47 | 49 | 37 | 28 | 48 | 40 | R42, R43 |
| `harrier-oss-v1-0.6b`, own fit | 1024 | 45.1% | 47 | 42 | 49 | 33 | 50 | 48 | R43 |
| `harrier-oss-v1-0.6b` + causal query instruction | 1024 | **47.8%** | 57 | 46 | 53 | 30 | 48 | 45 | R43 |

Noise floor between two runs of the same configuration: 1.7 points (R11). Differences
smaller than that are not reported as differences.

### Public-model rows: what you can reproduce today

Same 335 cases, thresholds, reranker and commit (`reasongraph` main `4f88153`, before the
bounded bridge), with the public [`Berk/causal-span-pointer-v2`](https://huggingface.co/Berk/causal-span-pointer-v2)
extractor and CPU embedders, run with [`eval/cause_at_1.py`](eval/cause_at_1.py):

| embedder | cause@1 | vs the v3 row above |
|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 30.4% | -2.6 |
| `multilingual-e5-large`, int8 ONNX | 40.0% | -2.4 |

The only difference between the two tables is the extractor. Details in
[`results/2026-09-17-public-cause-at-1.md`](results/2026-09-17-public-cause-at-1.md).

Why the embedder moves the number: on real data the small production embedder seeds the
gold cause directly in only 12% of questions, 74% of its top-5 seeds are noise from other
cases, and 71% of the answers it finds are reached by the walk alone (R44). A retrieval-
trained embedder seeds the gold cause 32% of the time and halves the walk-only share.

### Screening: does the embedder rank the cause near the effect at all?

Median cosine rank of the gold cause given its effect, 1,411 facts, no fine-tuning, and
the share of causes inside the top 25 (R31, R42, R43):

| embedder | top-25 |
|---|---|
| `harrier-oss-v1-0.6b` | 74% |
| `multilingual-e5-large-instruct` | 69% |
| `multilingual-e5-large` | 65% |
| `multilingual-e5-small` | 60% |
| `embeddinggemma-300m` | 58% |
| `Qwen3-Embedding-4B` | 55% |
| `voyage-4-nano` | 54% |
| `Qwen3-Embedding-0.6B` | 50% |
| `harrier-oss-v1-270m` | 43% |
| `granite-embedding-311m-r2` | 39% |
| `bge-m3` | 38% |
| `paraphrase-multilingual-MiniLM-L12-v2` | 31% |

Size is not what is being bought: `mpnet` (768d) and `bge-m3` (1024d) land beside the
384d production model. Only retrieval-trained families move (R32).

## Vague questions: can the walk start at all?

15 fictional incidents in a busy tenant, questions that carry no anchor entity, plus
surface-matching distractors (R44b, R45):

| metric | MiniLM | harrier |
|---|---|---|
| walk can start (gold incident seeded) | 73% | 100% |
| seed noise | 44% | 24% |
| gold root reaches the model | 33% | 40% |

60% of the hard vague questions still miss with the better embedder, so the residual is
downstream of the embedder. The cause was missing edges between the walk's levels, not the
context cap (R45). A walk-stage bridge that traces from every recalled fact fixes exactly
that: on the vague probe the gold root reaches the model 33% -> 73% (MiniLM) and 40% -> 87%
(harrier), and immediate-cause cause@1 on real-swap 335 goes 33.1% -> 50.7% (R46). It is
**not shipped**: on the busy standing graph the same unbounded fan-out builds spurious
cross-case bridges, root-cause recall at bi@6 drops 22.9% -> 17.9%, noise rises, and recall
is 5-20x slower (R46, R15). A bounded bridge is the next experiment; it must pass all four
rows, not the two it improves.

## What did not help

Each of these was run to completion and is written up in `results/`.

| idea | result | file |
|---|---|---|
| Hybrid seeds: word-trigram channel fused with cosine (RRF) | negative in every language; lexical seeds crowd the budget and cannot surface a root phrased unlike the question | W32 |
| Bigger embedder of the same family (mpnet 768d, bge-m3 1024d) | no better than the 384d model; the training objective is what matters | R32 |
| New embedder with the old thresholds | 32.8% vs 32.3% baseline: nothing. The re-fit is what pays (35.2%) | R32 |
| Separate forward/backward hop budgets | inert on this data: the total-hops cap always binds first | W37 |
| Entity surface-form normalisation and containment linking | no root-recall gain; containment linked across unrelated cases 79% of the time | (reasongraph README, Models) |
| Augmented training data for the causal extractor | 0.675 vs 0.70 F1 without it | causal-span-model |
| Walk-bridge with unbounded fan-out | +17.6 cause@1 and +40/+47 on the vague probe, but -5 root recall at bi@6, more noise, 5-20x latency on the busy graph; held | R46, R15 |
| Retraining the causal extractor on after/once/stems-from cues | recall on those cues unchanged (36% -> 37%, "after" still 0%) and CNC 0.705 -> 0.696; with the token gate off, the *old* model already extracted them. The gate was the bottleneck, not the pointer | R47, R50 |
| Synthetic French as a proxy for real French | synthetic 25% vs real 24% cause@1 looked equal, but a +15 e5 gain on synthetic became -8 on a realistic pool; real French is a genuine weak cell | R33, R35 |

## Reproducing

The evaluation scripts in [`eval/`](eval/) run against the installed `reasongraph`
package with the case files from `cases/`; no private paths. `eval/README.md` maps each
script to the experiment it came from. Two things to know before trusting a re-run:

- The rows above were measured with ReasonGraph Cloud's causal extractor (v3), which is
  not yet public. The scripts default to the open
  [`Berk/causal-span-pointer-v2`](https://huggingface.co/Berk/causal-span-pointer-v2);
  the v2 rows are the second table above; the gap is 2.4-2.6 points.
- Vector search in the PostgreSQL backend is approximate (HNSW); two builds of the same
  index agree on 48 of 80 cases unless results are ordered with a content tie-break
  (W34, W36). `reasongraph >= 0.7.31` ties by content, so re-runs are repeatable.

Each results file states the command, the git shas of the code, the case counts, wall
time, and what was not checked.

## Licence

Code and synthetic case sets: MIT. The `cases/*_real.jsonl` sets are derived from
Wikipedia and are CC BY-SA 4.0; see [`cases/README.md`](cases/README.md).
