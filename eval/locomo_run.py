"""LoCoMo (Maharana et al., ACL 2024) via ReasonGraph -- the agent-memory benchmark memory vendors quote.

Four cached stages so a re-judge never re-pays the answer model:
  build   ingest each conversation into its own scope (one fact per turn as "<speaker> (<date>): <text>",
          ChatExtractor on, causal extractor + reranker as prod), then recall the injected context for every
          question, timing the recall. Local, no LLM. Writes build.json (per-question context + gold +
          category + evidence + recall_ms + ingest stats).
  answer  read build.json, answer categories 1-4 with mem0's answer prompt on our recalled context,
          calling --answer-model -> answers.json. Needs an answer key.
  judge   read answers.json, score with mem0's CORRECT/WRONG judge prompt + --judge-model -> judge.json.
  report  read judge.json (+ answers.json), print and write the J score, per-category breakdown, token F1,
          mean injected-context tokens and p50 recall latency -> report.json + a Markdown table.

The answer and judge prompts and the 1-4 category set are mem0's, vendored verbatim in locomo_prompts.py, so
the retrieval is ours but the scoring is comparable to the numbers memory vendors publish (adversarial cat 5
is excluded, as mem0 does; J is the mean over categories 1-4 = 1540 questions).

Models are addressed as ``provider:model`` (provider in {openai, groq, ollama}); a bare name is inferred.
Keys come from the environment: OPENAI_API_KEY (openai), GROQ_API_KEY (groq), none for ollama (local).

Data: --data points at locomo10.json (github.com/snap-research/locomo; not redistributed here). Pre-registered:
no threshold tuning on the benchmark.

    python eval/locomo_run.py --stage build  --data <locomo10.json> --out <dir>
    OPENAI_API_KEY=... python eval/locomo_run.py --stage answer --data <locomo10.json> --out <dir> \
        --answer-model openai:gpt-4o-mini
    OPENAI_API_KEY=... python eval/locomo_run.py --stage judge  --data <locomo10.json> --out <dir> \
        --judge-model openai:gpt-4o
    python eval/locomo_run.py --stage report --out <dir>
"""
import argparse
import json
import os
import re
import string
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict

from locomo_prompts import (
    ANSWERER_MEMORY_LIMIT,
    CATEGORIES_TO_EVALUATE,
    CATEGORY_NAMES,
    JUDGE_SYSTEM_PROMPT,
    get_answer_generation_prompt,
    get_judge_prompt,
    get_judge_prompt_with_evidence,
    preprocess_answer,
)

CAUSAL_DEFAULT = "Berk/causal-span-pointer-v2"
TOKEN_GATE = "hf://Berk/causal-span-pointer-v2/token_gate"

# provider -> (base_url, api-key env var). ollama needs no key.
PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "ollama": (os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"), None),
}


# ---------------------------------------------------------------------------
# build stage (local, no LLM)
# ---------------------------------------------------------------------------

def conversation_facts(conv):
    """One fact per turn, "<speaker> (<session date>): <text>", in chronological order."""
    facts, i = [], 1
    while f"session_{i}" in conv:
        date = conv.get(f"session_{i}_date_time", "")
        for t in conv[f"session_{i}"]:
            facts.append(f"{t['speaker']} ({date}): {t['text']}")
        i += 1
    return facts


def stage_build(data_path, causal_model, out_dir):
    from reasongraph import ChatExtractor, MemoryLoop, ReasonGraph
    from reasongraph._extraction import CausalPointerExtractor

    from standing_row import RERANKER, embed_model

    samples = json.load(open(data_path, encoding="utf-8"))
    ext = CausalPointerExtractor(model=causal_model, gate_threshold=1.0,
                                 token_gate=TOKEN_GATE, token_gate_threshold=0.1)
    records, stats = [], []
    for s in samples:
        sid = str(s["sample_id"])
        facts = conversation_facts(s["conversation"])
        g = ReasonGraph(embed_model=embed_model(), rerank_model=RERANKER, causal_extractor=ext)
        g.initialize_sync()
        t = time.time()
        g.add_texts_sync(facts, scopes={sid}, extractor=ChatExtractor())
        ingest_s = round(time.time() - t, 1)
        for qa in s.get("qa", []):
            loop = MemoryLoop(g, session="q-" + sid, recall_scopes={sid})
            t = time.time()
            blk = loop.recall_sync(qa["question"])
            recall_ms = round((time.time() - t) * 1000, 1)
            ctx = [f["content"] for f in blk.facts]
            records.append({"sample": sid, "question": qa["question"], "gold": qa.get("answer"),
                            "evidence": qa.get("evidence"), "category": qa.get("category"),
                            "context": ctx, "n_ctx": len(ctx), "recall_ms": recall_ms})
        g.close_sync()
        stats.append({"sample": sid, "facts": len(facts), "ingest_s": ingest_s, "qa": len(s.get("qa", []))})
        print(f"  {sid}: {len(facts)} facts, ingest {ingest_s}s, {len(s.get('qa', []))} qa", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"embed": embed_model(), "causal_model": causal_model, "records": records, "stats": stats},
              open(os.path.join(out_dir, "build.json"), "w"), ensure_ascii=False, indent=1)
    total_q = sum(st["qa"] for st in stats)
    avg_ctx = sum(r["n_ctx"] for r in records) / max(len(records), 1)
    print(f"\nbuilt {len(stats)} conversations, {total_q} questions, mean injected facts/question {avg_ctx:.1f} "
          f"-> {out_dir}/build.json", flush=True)


# ---------------------------------------------------------------------------
# LoCoMo data helpers (reference date + evidence), keyed by sample id
# ---------------------------------------------------------------------------

def parse_locomo_date(date_str):
    """Parse a LoCoMo session date, e.g. '1:56 pm on 8 May, 2023'. Returns a datetime or None."""
    from datetime import datetime
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def reference_dates(data_path):
    """sample_id -> the last (most recent) session's date string, mem0's reference_date."""
    out = {}
    for s in json.load(open(data_path, encoding="utf-8")):
        conv = s["conversation"]
        dated = []
        for k in conv:
            m = re.match(r"^session_(\d+)$", k)
            if not m:
                continue
            dstr = conv.get(f"{k}_date_time", "")
            parsed = parse_locomo_date(dstr)
            key = (0, parsed) if parsed else (1, int(m.group(1)))
            dated.append((key, dstr))
        dated.sort(key=lambda x: (x[0][0], x[0][1] if x[0][0] == 1 else x[0][1].timestamp()))
        out[str(s["sample_id"])] = dated[-1][1] if dated else None
    return out


def evidence_lookup(data_path):
    """(sample_id, dia_id) -> a formatted evidence line, for the evidence-aware judge (mem0's format)."""
    lookup = {}
    for s in json.load(open(data_path, encoding="utf-8")):
        sid = str(s["sample_id"])
        conv = s["conversation"]
        session_dates = {k.replace("session_", "").replace("_date_time", ""): conv[k]
                         for k in conv if k.startswith("session_") and k.endswith("_date_time")}
        for k in conv:
            if not (k.startswith("session_") and not k.endswith("_date_time")):
                continue
            if not isinstance(conv[k], list):
                continue
            for turn in conv[k]:
                dia_id = turn.get("dia_id", "")
                if not dia_id:
                    continue
                m = re.match(r"D(\d+):", dia_id)
                date_suffix = ""
                if m:
                    sdate = session_dates.get(m.group(1), "")
                    if sdate:
                        date_suffix = f", said on {sdate}"
                lookup[(sid, dia_id)] = f'[{dia_id}{date_suffix}] {turn.get("speaker", "")}: "{turn.get("text", "")}"'
    return lookup


_FACT_RE = re.compile(r"^(?P<speaker>.+?) \((?P<date>[^)]+)\): (?P<text>.*)$", re.DOTALL)


def fact_to_result(fact):
    """Our recalled fact string -> the {memory, created_at} shape mem0's answer prompt expects.

    Keeps speaker attribution; created_at (ISO) lets the prompt sort chronologically and show a human date.
    """
    m = _FACT_RE.match(fact)
    if not m:
        return {"memory": fact, "created_at": ""}
    dt = parse_locomo_date(m.group("date"))
    created = dt.strftime("%Y-%m-%dT%H:%M:%S") if dt else ""
    return {"memory": f"{m.group('speaker')}: {m.group('text')}", "created_at": created}


def question_ids(records):
    """Deterministic per-question id, stable across runs given a fixed build.json order."""
    seen = defaultdict(int)
    ids = []
    for r in records:
        sid = r["sample"]
        ids.append(f"{sid}#{seen[sid]}")
        seen[sid] += 1
    return ids


# ---------------------------------------------------------------------------
# OpenAI-compatible chat client (stdlib only; works for openai / groq / ollama)
# ---------------------------------------------------------------------------

def parse_model(spec):
    """'provider:model' -> (provider, model). A bare name is inferred (ollama has 'name:size')."""
    head = spec.split(":", 1)[0]
    if head in PROVIDERS:
        return head, spec.split(":", 1)[1]
    if spec.startswith("gpt-4o") or spec.startswith("gpt-4") or spec.startswith("o1") or spec.startswith("o3"):
        return "openai", spec
    if "120b" in spec or "gpt-oss-120b" in spec:
        return "groq", spec
    if ":" in spec:  # ollama models look like 'gpt-oss:20b'
        return "ollama", spec
    raise SystemExit(f"cannot infer a provider for model '{spec}'; use provider:model (openai/groq/ollama)")


def make_client(spec):
    provider, model = parse_model(spec)
    base_url, key_env = PROVIDERS[provider]
    key = os.environ.get(key_env) if key_env else None
    if key_env and not key:
        raise SystemExit(f"{key_env} is not set (needed for provider '{provider}', model '{model}')")
    return {"provider": provider, "model": model, "base_url": base_url.rstrip("/"), "key": key}


def chat_completion(client, system, user, json_mode=False, temperature=0.0, retries=5):
    payload = {"model": client["model"], "temperature": temperature,
               "messages": ([{"role": "system", "content": system}] if system else [])
               + [{"role": "user", "content": user}]}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if client["key"]:
        headers["Authorization"] = f"Bearer {client['key']}"
    url = client["base_url"] + "/chat/completions"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 529) and attempt < retries - 1:
                time.sleep(min(2 ** attempt, 30))
                continue
            detail = e.read().decode("utf-8", "replace")[:300] if hasattr(e, "read") else ""
            raise SystemExit(f"answer/judge call failed ({e.code}): {detail}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            if attempt < retries - 1:
                time.sleep(min(2 ** attempt, 30))
                continue
            raise SystemExit(f"answer/judge call failed: {e}") from e
    raise SystemExit(f"answer/judge call failed after {retries} tries: {last}")


def parse_judge_json(text):
    """Return {'label','reasoning'} from a judge reply, tolerant of stray prose around the JSON."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    up = (text or "").upper()
    return {"label": "CORRECT" if "CORRECT" in up and "WRONG" not in up else "WRONG", "reasoning": "unparsed"}


# ---------------------------------------------------------------------------
# answer stage
# ---------------------------------------------------------------------------

def load_json(path, default):
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return default


def stage_answer(build_path, data_path, model_spec, out_dir, limit=None):
    build = load_json(build_path, None)
    if build is None:
        raise SystemExit(f"no build.json at {build_path}; run --stage build first")
    records = build["records"]
    ids = question_ids(records)
    refs = reference_dates(data_path)
    client = make_client(model_spec)
    ans_path = os.path.join(out_dir, "answers.json")
    answers = load_json(ans_path, {})
    todo = [(qid, r) for qid, r in zip(ids, records) if r["category"] in CATEGORIES_TO_EVALUATE
            and qid not in answers]
    if limit:
        todo = todo[:limit]
    print(f"answering {len(todo)} questions ({model_spec}); {len(answers)} already cached", flush=True)
    for n, (qid, r) in enumerate(todo, 1):
        search_results = [fact_to_result(c) for c in r["context"][:ANSWERER_MEMORY_LIMIT]]
        prompt = get_answer_generation_prompt(r["question"], search_results,
                                              reference_date=refs.get(r["sample"]))
        gen = chat_completion(client, system="", user=prompt)
        if "ANSWER:" in gen:
            gen = gen.rsplit("ANSWER:", 1)[-1].strip()
        answers[qid] = {"sample": r["sample"], "category": r["category"], "question": r["question"],
                        "gold": r["gold"], "evidence": r.get("evidence") or [],
                        "generated": gen, "n_ctx": r["n_ctx"], "prompt_chars": len(prompt)}
        if n % 20 == 0 or n == len(todo):
            json.dump(answers, open(ans_path, "w"), ensure_ascii=False, indent=1)
            print(f"  {n}/{len(todo)} answered", flush=True)
    json.dump(answers, open(ans_path, "w"), ensure_ascii=False, indent=1)
    print(f"answers -> {ans_path} ({len(answers)} total)", flush=True)


# ---------------------------------------------------------------------------
# judge stage
# ---------------------------------------------------------------------------

def stage_judge(out_dir, data_path, model_spec, use_evidence=True, limit=None):
    ans_path = os.path.join(out_dir, "answers.json")
    answers = load_json(ans_path, None)
    if answers is None:
        raise SystemExit(f"no answers.json at {ans_path}; run --stage answer first")
    ev = evidence_lookup(data_path) if use_evidence else {}
    client = make_client(model_spec)
    judge_path = os.path.join(out_dir, "judge.json")
    judged = load_json(judge_path, {})
    todo = [(qid, a) for qid, a in answers.items() if qid not in judged]
    if limit:
        todo = todo[:limit]
    print(f"judging {len(todo)} answers ({model_spec}); {len(judged)} already cached", flush=True)
    for n, (qid, a) in enumerate(todo, 1):
        cat = a["category"]
        gold = preprocess_answer(cat, str(a["gold"]))
        ev_ctx = ""
        if use_evidence:
            ev_ctx = "\n".join(ev[(a["sample"], d)] for d in a.get("evidence", []) if (a["sample"], d) in ev)
        if ev_ctx:
            prompt = get_judge_prompt_with_evidence(cat, a["question"], gold, a["generated"], ev_ctx)
        else:
            prompt = get_judge_prompt(cat, a["question"], gold, a["generated"])
        raw = parse_judge_json(chat_completion(client, JUDGE_SYSTEM_PROMPT, prompt, json_mode=True))
        correct = str(raw.get("label", "")).upper() == "CORRECT"
        judged[qid] = {"category": cat, "label": "CORRECT" if correct else "WRONG",
                       "score": 1.0 if correct else 0.0, "reasoning": raw.get("reasoning", "")}
        if n % 20 == 0 or n == len(todo):
            json.dump(judged, open(judge_path, "w"), ensure_ascii=False, indent=1)
            print(f"  {n}/{len(todo)} judged", flush=True)
    json.dump(judged, open(judge_path, "w"), ensure_ascii=False, indent=1)
    print(f"judgments -> {judge_path} ({len(judged)} total)", flush=True)


# ---------------------------------------------------------------------------
# report stage
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}


def _normalize(s):
    s = (s or "").lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return [w for w in s.split() if w not in _ARTICLES]


def token_f1(pred, gold):
    p, g = _normalize(pred), _normalize(gold)
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def _count_tokens(text, encoder):
    if encoder is not None:
        return len(encoder.encode(text))
    return int(len(text.split()) * 1.3)  # rough estimate when tiktoken is absent


def stage_report(out_dir, build_path=None):
    answers = load_json(os.path.join(out_dir, "answers.json"), {})
    judged = load_json(os.path.join(out_dir, "judge.json"), {})
    if not judged:
        raise SystemExit(f"no judge.json in {out_dir}; run --stage judge first")
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoder = None

    by_cat_score = defaultdict(list)
    by_cat_f1 = defaultdict(list)
    ctx_tokens = []
    for qid, j in judged.items():
        cat = j["category"]
        by_cat_score[cat].append(j["score"])
        a = answers.get(qid)
        if a:
            by_cat_f1[cat].append(token_f1(a["generated"], preprocess_answer(cat, str(a["gold"]))))

    # mean injected-context tokens per question (over the answered 1-4 set), from build.json context
    recall_ms = []
    if build_path and os.path.exists(build_path):
        build = load_json(build_path, {})
        ids = question_ids(build["records"])
        idmap = dict(zip(ids, build["records"]))
        for qid, j in judged.items():
            r = idmap.get(qid)
            if not r:
                continue
            block = "\n".join(fact_to_result(c)["memory"] for c in r["context"])
            ctx_tokens.append(_count_tokens(block, encoder))
            if r.get("recall_ms") is not None:
                recall_ms.append(r["recall_ms"])

    cats = sorted(by_cat_score)
    all_scores = [s for c in cats for s in by_cat_score[c]]
    all_f1 = [f for c in cats for f in by_cat_f1[c]]
    j_overall = 100 * sum(all_scores) / len(all_scores) if all_scores else 0.0

    def p50(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None

    report = {
        "out_dir": out_dir,
        "n_judged": len(judged),
        "J_overall": round(j_overall, 1),
        "token_f1_overall": round(100 * sum(all_f1) / len(all_f1), 1) if all_f1 else None,
        "per_category": {CATEGORY_NAMES.get(c, str(c)): {
            "n": len(by_cat_score[c]),
            "J": round(100 * sum(by_cat_score[c]) / len(by_cat_score[c]), 1),
            "token_f1": round(100 * sum(by_cat_f1[c]) / len(by_cat_f1[c]), 1) if by_cat_f1[c] else None,
        } for c in cats},
        "mean_ctx_tokens": round(sum(ctx_tokens) / len(ctx_tokens)) if ctx_tokens else None,
        "recall_ms_p50": p50(recall_ms),
        "tiktoken": encoder is not None,
    }
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), ensure_ascii=False, indent=1)

    lines = [f"J (categories 1-4, n={len(all_scores)}): {report['J_overall']}",
             f"token F1: {report['token_f1_overall']}",
             f"mean injected-context tokens/question: {report['mean_ctx_tokens']}"
             f"{'' if encoder else ' (whitespace estimate; tiktoken not installed)'}",
             f"p50 recall latency (ms): {report['recall_ms_p50']}", "",
             "| category | n | J | token F1 |", "|---|---|---|---|"]
    for c in cats:
        pc = report["per_category"][CATEGORY_NAMES.get(c, str(c))]
        lines.append(f"| {CATEGORY_NAMES.get(c, c)} | {pc['n']} | {pc['J']} | {pc['token_f1']} |")
    print("\n".join(lines))
    print(f"\nreport -> {os.path.join(out_dir, 'report.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["build", "answer", "judge", "report"])
    ap.add_argument("--data", help="path to locomo10.json (build/answer/judge stages)")
    ap.add_argument("--causal-model", default=CAUSAL_DEFAULT)
    ap.add_argument("--answer-model", default="groq:openai/gpt-oss-120b")
    ap.add_argument("--judge-model", default="openai:gpt-4o")
    ap.add_argument("--no-evidence", action="store_true", help="judge without the evidence-aware rule")
    ap.add_argument("--limit", type=int, help="cap questions (smoke tests)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build_path = os.path.join(args.out, "build.json")
    if args.stage == "build":
        if not args.data:
            ap.error("--data is required for the build stage")
        stage_build(args.data, args.causal_model, args.out)
    elif args.stage == "answer":
        if not args.data:
            ap.error("--data is required for the answer stage (reference dates)")
        stage_answer(build_path, args.data, args.answer_model, args.out, limit=args.limit)
    elif args.stage == "judge":
        if not args.data:
            ap.error("--data is required for the judge stage (evidence lookup)")
        stage_judge(args.out, args.data, args.judge_model, use_evidence=not args.no_evidence, limit=args.limit)
    elif args.stage == "report":
        stage_report(args.out, build_path=build_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
