# eval

Standing-row scripts, ported from the lab repository to run against the **installed** `reasongraph` package.
No absolute paths and no `sys.path` edits: the scripts import `reasongraph`, read the case sets from `../cases`,
take the embedder from `REASONGRAPH_EMBED_MODEL`, and take the span-pointer model as `--causal-model` (default the
public `Berk/causal-span-pointer-v2`).

## Scripts

| script | measures |
|---|---|
| `cause_at_1.py` | **cause@1** on the real-swap 335 (the headline metric): is the gold immediate cause among the recalled facts, per language |
| `standing_row.py` | the standing busy-tenant row (context / root / non-chain recall, ask-twice); also the shared case-loading and metric helpers |
| `r13_floor.py` | case-set helpers (`load_cases`, `mode_of`, `primary_rel`) |
| `onnx_embedder.py` | an int8 ONNX embedder for CPU (`REASONGRAPH_EMBED_MODEL=onnx-int8:<hf-id>`); exports + caches on first use |
| `locomo_run.py` | **LoCoMo J score** (the number memory vendors quote), in four cached stages: `build` (recall our context per question, local), `answer`, `judge`, `report`. `locomo_prompts.py` is mem0's answer + CORRECT/WRONG judge vendored verbatim (Apache-2.0), so retrieval is ours and scoring is comparable (categories 1-4, adversarial 5 excluded) |

Lab -> bench name mapping: the lab's per-experiment cause@1 scripts (`r42c`/`r43d`) are unified here as
`cause_at_1.py`; `standing_row.py` and `r13_floor.py` keep their lab names.

## Run

```bash
pip install reasongraph            # plus torch, sentence-transformers, onnxruntime (+ onnxscript for onnx-int8)
# production baseline (MiniLM, CPU)
REASONGRAPH_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    python eval/cause_at_1.py --out results/cause_minilm.json
# e5-large int8 (CPU), its own thresholds
REASONGRAPH_EMBED_MODEL=onnx-int8:intfloat/multilingual-e5-large \
    python eval/cause_at_1.py --span-link 0.975 --dedup 0.949 --min-score 0.75 --min-ratio 0.90 \
    --out results/cause_e5large_int8.json
```

The default embedder and thresholds are the production MiniLM configuration. Each embedder uses its own re-fitted
thresholds (see the top-level README leaderboard for the values).

## LoCoMo

`locomo_run.py` runs LoCoMo (Maharana et al., ACL 2024) through ReasonGraph and scores it with mem0's public
J-score harness, so the number is comparable to the ones memory vendors publish. Get `locomo10.json` from
github.com/snap-research/locomo (not redistributed here). The four stages cache their outputs under `--out`, so a
re-judge never re-pays the answer model. Models are `provider:model` (openai / groq / ollama); keys come from
`OPENAI_API_KEY` / `GROQ_API_KEY` (ollama is local, no key).

```bash
# 1. build: recall our injected context for every question (local, no LLM, ~34 min on a GPU box)
python eval/locomo_run.py --stage build  --data locomo10.json --out runs/locomo
# 2. answer categories 1-4 with the answer model
OPENAI_API_KEY=... python eval/locomo_run.py --stage answer --data locomo10.json --out runs/locomo \
    --answer-model openai:gpt-4o-mini
# 3. judge with mem0's CORRECT/WRONG prompt (evidence-aware by default)
OPENAI_API_KEY=... python eval/locomo_run.py --stage judge  --data locomo10.json --out runs/locomo \
    --judge-model openai:gpt-4o
# 4. report the J score, per-category breakdown, token F1, mean context tokens, p50 recall latency
python eval/locomo_run.py --stage report --out runs/locomo    # add `uv run --with tiktoken` for exact token counts
```
