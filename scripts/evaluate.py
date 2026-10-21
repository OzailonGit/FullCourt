"""Evaluate the three retrieval methods against graded relevance labels.

Two subcommands:

  python scripts/evaluate.py pool
      Runs every query in eval/queries.json with all three methods and writes the union of each
      method's top results to eval/label_pool.csv (TREC-style pooling) with an empty `grade`
      column, keeping any grades already present in eval/labels.csv.

  python scripts/evaluate.py run
      Reads eval/labels.csv (grade 0 = not relevant, 1 = partially relevant, 2 = highly relevant)
      and reports Recall@10, nDCG@10, judged@10 and per-stage latency for each method, writing
      eval/results.json and eval/results.md.

Metrics
  Recall@10 = |relevant in top 10| / |relevant|, relevant meaning grade >= 1.
  nDCG@10   = DCG@10 / IDCG@10 with gain 2^grade - 1 and discount log2(rank + 1); the ideal
              ranking uses every labeled player for the query.
  judged@10 = share of the top 10 that has a label. Unlabeled players count as grade 0, so low
              judged@10 means the metric is less trustworthy for that method.
Queries with no relevant labels are excluded from the averages and listed separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pipeline  # noqa: E402
from src.retrieval.hybrid import format_stat  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
METHODS = ["stats", "semantic", "hybrid"]
K = 10
POOL_DEPTH = 15


def load_queries() -> list[dict]:
    return json.loads((EVAL_DIR / "queries.json").read_text(encoding="utf-8"))


def load_labels(path: Path) -> dict[str, dict[str, int]]:
    labels: dict[str, dict[str, int]] = {}
    if not path.exists():
        return labels
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["grade"].strip() != "":
                labels.setdefault(row["query_id"], {})[row["player_id"]] = int(row["grade"])
    return labels


def recall_at_k(ranked: list[str], grades: dict[str, int], k: int = K) -> float | None:
    relevant = {p for p, g in grades.items() if g >= 1}
    if not relevant:
        return None
    return len(relevant & set(ranked[:k])) / len(relevant)


def ndcg_at_k(ranked: list[str], grades: dict[str, int], k: int = K) -> float | None:
    def dcg(gains: list[int]) -> float:
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))

    ideal = dcg(sorted(grades.values(), reverse=True))
    if ideal == 0:
        return None
    return dcg([grades.get(p, 0) for p in ranked]) / ideal


def run_all(queries: list[dict], depth: int) -> dict[str, dict[str, object]]:
    engine = pipeline.load_engine()
    settings = pipeline.default_settings()
    settings.limit = depth
    llm, _ = pipeline.load_llm()
    runs: dict[str, dict[str, object]] = {}
    for q in queries:
        for method in METHODS:
            output = pipeline.run(q["question"], engine, method, settings, llm, generate=False)
            runs.setdefault(q["id"], {})[method] = output
    return runs


def cmd_pool() -> None:
    queries = load_queries()
    existing = load_labels(EVAL_DIR / "labels.csv")
    runs = run_all(queries, POOL_DEPTH)
    rows = []
    for q in queries:
        pooled: dict[str, dict] = {}
        for method in METHODS:
            for r in runs[q["id"]][method].result.results:
                entry = pooled.setdefault(r.player_id, {"player": r, "methods": []})
                entry["methods"].append(f"{method}#{r.rank}")
        for pid, entry in sorted(pooled.items(), key=lambda kv: kv[1]["player"].player_name):
            r = entry["player"]
            s = r.stats
            rows.append({
                "query_id": q["id"],
                "question": q["question"],
                "player_id": pid,
                "player_name": r.player_name,
                "position": s.get("position"),
                "ppg": format_stat("points_per_game", s.get("points_per_game")),
                "apg": format_stat("assists_per_game", s.get("assists_per_game")),
                "ts": format_stat("true_shooting_pct", s.get("true_shooting_pct")),
                "usg": format_stat("usage_rate", s.get("usage_rate")),
                "found_by": " ".join(entry["methods"]),
                "grade": existing.get(q["id"], {}).get(pid, ""),
            })
    out = EVAL_DIR / "label_pool.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} pooled judgments for {len(queries)} queries to {out}")


def cmd_run(labels_path: Path) -> None:
    queries = load_queries()
    labels = load_labels(labels_path)
    runs = run_all(queries, K)

    per_query, timings = [], {m: {} for m in METHODS}
    for q in queries:
        grades = labels.get(q["id"], {})
        row = {"query_id": q["id"], "question": q["question"], "intent": q["intent"]}
        for method in METHODS:
            output = runs[q["id"]][method]
            ranked = [r.player_id for r in output.result.results]
            row[method] = {
                "recall@10": recall_at_k(ranked, grades),
                "ndcg@10": ndcg_at_k(ranked, grades),
                "judged@10": (sum(1 for p in ranked[:K] if p in grades) / min(K, len(ranked))) if ranked else None,
                "returned": len(ranked),
            }
            for stage, ms in output.result.timings_ms.items():
                timings[method].setdefault(stage, []).append(ms)
        per_query.append(row)

    summary = {}
    for method in METHODS:
        summary[method] = {}
        for metric in ("recall@10", "ndcg@10", "judged@10"):
            values = [r[method][metric] for r in per_query if r[method][metric] is not None]
            summary[method][metric] = round(statistics.mean(values), 3) if values else None
            summary[method][f"{metric}_n"] = len(values)
        summary[method]["latency_ms_median"] = {s: round(statistics.median(v), 1) for s, v in timings[method].items()}

    unlabeled = [q["id"] for q in queries if not any(g >= 1 for g in labels.get(q["id"], {}).values())]
    report = {"labels_file": str(labels_path), "k": K, "summary": summary, "per_query": per_query, "queries_without_relevant_labels": unlabeled}
    (EVAL_DIR / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (EVAL_DIR / "results.md").write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def render_markdown(report: dict) -> str:
    lines = ["| Method | Recall@10 | nDCG@10 | judged@10 | queries | median latency (ms) |", "|---|---|---|---|---|---|"]
    for method, s in report["summary"].items():
        latency = ", ".join(f"{k} {v}" for k, v in s["latency_ms_median"].items())
        lines.append(f"| {method} | {_fmt(s['recall@10'])} | {_fmt(s['ndcg@10'])} | {_fmt(s['judged@10'])} | {s['recall@10_n']} | {latency} |")
    lines += ["", "| Query | Intent | " + " | ".join(f"{m} R@10 / nDCG@10" for m in METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
    for r in report["per_query"]:
        cells = " | ".join(f"{_fmt(r[m]['recall@10'])} / {_fmt(r[m]['ndcg@10'])}" for m in METHODS)
        lines.append(f"| {r['query_id']} | {r['intent']} | {cells} |")
    if report["queries_without_relevant_labels"]:
        lines += ["", "Queries without relevant labels (excluded from averages): " + ", ".join(report["queries_without_relevant_labels"])]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("pool")
    run = sub.add_parser("run")
    run.add_argument("--labels", type=Path, default=EVAL_DIR / "labels.csv")
    args = parser.parse_args()
    if args.command == "pool":
        cmd_pool()
    else:
        cmd_run(args.labels)


if __name__ == "__main__":
    main()
