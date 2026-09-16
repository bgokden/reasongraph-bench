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
