# Case sets

Every case is a multi-source *why*-question: a handful of facts that were written
independently (`sessions`), a question, the gold chain of facts an answer must walk,
and the gold answer. The metric that matters is **cause@1**: does the fact ranked first
carry the gold cause.

| file | n | languages | origin | licence |
|---|---|---|---|---|
| `causal_cases.jsonl` | 40 | en (32), de, nl, es, fr, tr | hand-reviewed; ships in `reasongraph/tests/data` | MIT |
| `causal_cases_batch2.jsonl` | 40 | en (28), de, es, fr, tr, nl | 8 domains x 5, generated with a local open model, reviewed | MIT |
| `causal_cases_batch3.jsonl` | 60 | de, es, fr (20 each) | generated with a local open model, reviewed | MIT |
| `causal_cases_w5_trnl.jsonl` | 40 | tr, nl (20 each) | generated with a local open model, reviewed | MIT |
| `causal_cases_r11.jsonl` | 161 | nl 36, es 33, de 29, fr 29, tr 19, en 15 | generated with a local open model; each case names its distractor mode in `notes` | MIT |
| `causal_cases_fr_real.jsonl` | 46 | fr | verbatim sentences from fr.wikipedia incident articles (Ariane 501, Piper Alpha, Tchernobyl, ...); the source URL is in every `notes` field; questions written by a model | CC BY-SA 4.0 |
| `causal_cases_en_real.jsonl` | 10 | en | verbatim sentences from en.wikipedia (Knight Capital, Northeast blackout of 2003, ...); source in `notes` | CC BY-SA 4.0 |
| `causal_cases_es_real.jsonl` | 19 | es | verbatim sentences from es.wikipedia (Chernobyl, Bhopal, ...); source in `notes` | CC BY-SA 4.0 |
| `vague_followups.jsonl` | 15 | en | fictional ops incidents in a busy tenant, anchor-free questions plus surface-matching distractors | MIT |

The synthetic sets name no real organisations and contain no travel content (checked by scan).
The three `*_real` sets are derivative works of Wikipedia articles and are therefore
released under **CC BY-SA 4.0**, with attribution to the article named in each record.
Everything else in this directory is MIT like the code.

## Standing rows

- **Standing 341**: the first five files together.
- **Real-swap 335**: the same pool with its 52 synthetic French cases replaced by the 46
  real French cases. This is the row the leaderboard in the top-level README quotes,
  because the synthetic French cell turned out to be easier than real French prose
  (see `results/`).
- **Vague probe**: `vague_followups.jsonl` on its own, to measure whether a walk can
  start at all when the question carries no anchor entity.

## Record format

```json
{"id": "...", "lang": "en", "domain": "software",
 "sessions": {"SessionA": ["fact 1", "distractor"], "SessionB": ["fact 2"], "SessionC": ["fact 3"]},
 "question": "Why did ...?",
 "gold_chain": ["fact 1", "fact 2", "fact 3"],
 "gold_answer": "...",
 "notes": "distractor '...' shares '...'"}
```

Each session is one independent source; the gold chain crosses sessions on purpose.

`vague_followups.jsonl` records carry `incident` (the facts), `question`, and `distractors`.
