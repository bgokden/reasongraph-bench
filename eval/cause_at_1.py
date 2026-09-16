"""cause@1 on the real-swap 335 -- the headline metric. For each case, ingest the facts, ask the why-question,
ask it again (the second ask is what an agent sees), and check whether the gold immediate cause is among the
recalled facts. Reported per language.

Everything is held fixed except the embedder and its four thresholds, so the number isolates the embedder:
  - causal extractor: --causal-model (default the public Berk/causal-span-pointer-v2) + its token gate
  - reranker: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
  - embedder: REASONGRAPH_EMBED_MODEL (a sentence-transformers id, or "onnx-int8:<id>" for a CPU int8 build)
  - thresholds: --span-link / --dedup / --min-score / --min-ratio (re-fit per embedder)

Case files come from ../cases: the standing pool minus synthetic French, plus the real French set (the real-swap).

    # production baseline (MiniLM)
    REASONGRAPH_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        python eval/cause_at_1.py --out cause_minilm.json
    # e5-large, its own thresholds
    REASONGRAPH_EMBED_MODEL=intfloat/multilingual-e5-large \
        python eval/cause_at_1.py --span-link 0.975 --dedup 0.949 --min-score 0.75 --min-ratio 0.90 \
        --out cause_e5large.json
"""
import argparse
import collections
import json
import os

from reasongraph import MemoryLoop, ReasonGraph
from reasongraph._extraction import CausalPointerExtractor

from standing_row import ANSWER, CASES_DIR, CORPUS, RERANKER, _norm, embed_model
from r13_floor import load_cases

LANGS = ("en", "de", "es", "fr", "nl", "tr")


def _has(target, texts):
    t = _norm(target)
    return any(t == x or t in x or x in t for x in texts)


def resolve_embedder(spec):
    """A sentence-transformers id (reasongraph applies e5-style prefixes by name), or an int8 ONNX build."""
    if spec.startswith("onnx-int8:"):
        from onnx_embedder import OnnxInt8Embedder
        return OnnxInt8Embedder(spec.split(":", 1)[1])
    return spec


def run(cases, embed, ext, const):
    g = ReasonGraph(embed_model=embed, rerank_model=RERANKER, causal_extractor=ext,
                    span_link_threshold=const["span_link_threshold"])
    g.initialize_sync()
    pushed = [x for c in cases for fs in c["sessions"].values() for x in fs]
    g.add_texts_sync(pushed, scopes={CORPUS}, dedup_threshold=const["dedup_threshold"], dedup_entity_gate=True)
    per = collections.defaultdict(lambda: {"n": 0, "hit": 0})
    for c in cases:
        lang = c.get("lang", "en")
        want = c["gold_chain"][-2]
        sess = "s-" + c["id"]
        loop = MemoryLoop(g, session=sess, recall_scopes={CORPUS, sess},
                          min_score=const["min_score"], min_ratio=const["min_ratio"])
        loop.recall_sync(c["question"])
        loop.observe_sync(user=c["question"], assistant=ANSWER)
        facts = [_norm(f["content"]) for f in loop.recall_sync(c["question"]).facts]
        per[lang]["n"] += 1
        per[lang]["hit"] += int(_has(want, facts))
    g.close_sync()
    alln = sum(v["n"] for v in per.values())
    allhit = sum(v["hit"] for v in per.values())
    row = {"ALL": round(allhit / alln, 3)}
    for lg in LANGS:
        v = per.get(lg)
        row[lg] = round(v["hit"] / v["n"], 3) if v and v["n"] else None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--causal-model", default="Berk/causal-span-pointer-v2",
                    help="span-pointer model (public hf id by default; a local dir also works)")
    ap.add_argument("--token-gate", default="hf://Berk/causal-span-pointer-v2/token_gate")
    ap.add_argument("--real", default=os.path.join(CASES_DIR, "causal_cases_fr_real.jsonl"))
    ap.add_argument("--span-link", type=float, default=0.92)
    ap.add_argument("--dedup", type=float, default=0.88)
    ap.add_argument("--min-score", type=float, default=0.1)
    ap.add_argument("--min-ratio", type=float, default=0.45)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    const = {"span_link_threshold": args.span_link, "dedup_threshold": args.dedup,
             "min_score": args.min_score, "min_ratio": args.min_ratio}

    standing = [c for c in load_cases() if len(c.get("gold_chain", [])) >= 2]
    real_fr = [json.loads(l) for l in open(args.real, encoding="utf-8") if l.strip()]
    cases = [c for c in standing if c.get("lang") != "fr"] + real_fr
    print(f"real-swap {len(cases)} cases (real fr {len(real_fr)}) | embed={embed_model()} "
          f"| causal={args.causal_model} | constants={const}", flush=True)

    ext = CausalPointerExtractor(model=args.causal_model, gate_threshold=1.0,
                                 token_gate=args.token_gate, token_gate_threshold=0.1)
    row = run(cases, resolve_embedder(embed_model()), ext, const)
    out = {"embed": embed_model(), "causal_model": args.causal_model, "constants": const, "cause_at_1": row}
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\ncause@1 ALL {row['ALL']:.1%} | " + " ".join(f"{lg} {row[lg]:.0%}" for lg in LANGS if row[lg] is not None),
          flush=True)
    print(f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
