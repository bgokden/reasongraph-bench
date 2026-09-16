"""Standing busy-tenant row: all standing cases in one tenant, each question asked twice (the second ask is what
users see), reporting context / root / non-chain recall per language. Shared helpers (case loading, metrics) are
imported by the other scripts in this directory.

The embedder comes from REASONGRAPH_EMBED_MODEL (a sentence-transformers id); the reranker and causal extractor
are ReasonGraph's defaults unless overridden. Case files live in ../cases.

    REASONGRAPH_EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
        python eval/standing_row.py --out standing.json
"""
import argparse
import collections
import json
import os
import re

from reasongraph import MemoryLoop, ReasonGraph

CASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cases")
STANDING_FILES = ["causal_cases.jsonl", "causal_cases_batch2.jsonl", "causal_cases_batch3.jsonl",
                  "causal_cases_w5_trnl.jsonl", "causal_cases_r11.jsonl"]
RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CORPUS = "corpus"
ANSWER = "I don't have that on record yet."
DEFAULT_EMBED = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def embed_model():
    return os.environ.get("REASONGRAPH_EMBED_MODEL") or DEFAULT_EMBED


def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _contains(a, b):
    return a == b or a in b or b in a


def read_cases(filenames):
    cases = []
    for name in filenames:
        path = os.path.join(CASES_DIR, name)
        try:
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    cases.append(json.loads(line))
        except FileNotFoundError:
            print(f"  (missing {path})", flush=True)
    return cases


def load_cases():
    return read_cases(STANDING_FILES)


def metrics(block, gold_chain, sess):
    gold = {_norm(f) for f in gold_chain}
    root = _norm(gold_chain[0])
    inj = [_norm(f["content"]) for f in block.facts]
    ctx = sum(1 for g in gold if any(_contains(g, i) for i in inj)) / len(gold)
    root_hit = int(any(_contains(root, i) for i in inj))
    non_chain = sum(1 for i in inj if not any(_contains(i, g) for g in gold))
    return ctx, root_hit, non_chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--span-link-threshold", type=float, default=0.85)
    args = ap.parse_args()
    cases = load_cases()
    print(f"{len(cases)} cases {dict(collections.Counter(c.get('lang', 'en') for c in cases))} "
          f"| embed={embed_model()} | span_link_threshold={args.span_link_threshold}", flush=True)

    pushed = [x for c in cases for fs in c["sessions"].values() for x in fs]
    g = ReasonGraph(embed_model=embed_model(), rerank_model=RERANKER, span_link_threshold=args.span_link_threshold)
    g.initialize_sync()
    g.add_texts_sync(pushed, scopes={CORPUS}, dedup_threshold=0.88, dedup_entity_gate=True)
    rows = []
    for c in cases:
        sess = "s-" + c["id"]
        loop = MemoryLoop(g, session=sess, recall_scopes={CORPUS, sess}, min_score=0.1, min_ratio=0.45)
        loop.recall_sync(c["question"])
        loop.observe_sync(user=c["question"], assistant=ANSWER)
        b2 = loop.recall_sync(c["question"])
        c2, r2, nc2 = metrics(b2, c["gold_chain"], sess)
        rows.append({"id": c["id"], "lang": c.get("lang", "en"), "ctx2": c2, "root2": r2, "nc2": nc2})
    g.close_sync()

    def avg(rs, k):
        return sum(r[k] for r in rs) / len(rs) if rs else 0.0

    out = {}
    print(f"\n  {'lang':<5s} {'n':>3s} | {'ctx2':>5s} {'root2':>6s} {'nc2':>5s}")
    for lang in ("ALL", "en", "de", "es", "fr", "nl", "tr"):
        rs = rows if lang == "ALL" else [r for r in rows if r["lang"] == lang]
        if not rs:
            continue
        out[lang] = {k: avg(rs, k) for k in ("ctx2", "root2", "nc2")}
        out[lang]["n"] = len(rs)
        print(f"  {lang:<5s} {len(rs):>3d} | {avg(rs, 'ctx2'):>4.0%} {avg(rs, 'root2'):>5.0%} {avg(rs, 'nc2'):>5.2f}")
    json.dump({"per_lang": out, "rows": rows}, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
