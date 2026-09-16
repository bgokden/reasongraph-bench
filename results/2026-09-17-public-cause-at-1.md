# Reproducible cause@1 with the public model, on CPU

The two CPU-deployable embedders, measured end to end with the **public** span-pointer model
[`Berk/causal-span-pointer-v2`](https://huggingface.co/Berk/causal-span-pointer-v2) and its token gate, so the
numbers here are what anyone can reproduce from this repository. cause@1 on the real-swap 335 (the standing pool
with the synthetic French cell replaced by real French prose).

reasongraph `main` at `4f88153` (before the bounded-walk-bridge change is merged; every number below is on the
pre-bridge recall path). Reranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, causal extractor
`Berk/causal-span-pointer-v2` + its token gate at 0.1, pointer's own sentence gate off. Each embedder uses its own
re-fitted thresholds.

| embedder | dim | thresholds (span_link / dedup / min_score / min_ratio) | ALL | en | de | es | fr | nl | tr |
|---|---|---|---|---|---|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 0.92 / 0.88 / 0.10 / 0.45 | 30.4% | 37 | 29 | 28 | 20 | 38 | 24 |
| `multilingual-e5-large` int8 (ONNX, CPU) | 1024 | 0.975 / 0.949 / 0.75 / 0.90 | 40.0% | 39 | 47 | 33 | 17 | 53 | 48 |

The retrieval-trained embedder is +9.6 points over the production MiniLM (30.4 -> 40.0), and int8 quantization
keeps it CPU-deployable at that quality. The noise floor between two runs of one configuration is 1.7 points;
per-language cells are small-n (roughly 45-60 cases outside French, 46 real French) and move more.

## A note on the model version

The leaderboard in the top-level README quotes cause@1 with the lab's newer pointer (v3), which is ~3 points
higher than the public v2 measured here (MiniLM 33.0 vs 30.4, e5-large 42.4 vs 40.0 int8). The gap is entirely the
span-pointer model -- same cases, thresholds, embedder, reranker and reasongraph commit. So the README numbers are
the internal-model figures; the numbers in THIS file are the ones reproducible today from the public repository.
When the newer pointer is published the two will converge.

## Reproduce

```bash
pip install reasongraph            # + torch, sentence-transformers, onnxruntime, onnxscript
REASONGRAPH_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
    python eval/cause_at_1.py --out results/cause_minilm.json
REASONGRAPH_EMBED_MODEL=onnx-int8:intfloat/multilingual-e5-large \
    python eval/cause_at_1.py --span-link 0.975 --dedup 0.949 --min-score 0.75 --min-ratio 0.90 \
    --out results/cause_e5large_int8.json
```

Raw outputs: `results/cause_minilm.json`, `results/cause_e5large_int8.json`.
