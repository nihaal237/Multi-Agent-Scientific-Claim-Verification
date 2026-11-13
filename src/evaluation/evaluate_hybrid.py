"""evaluate the hybrid system against the same 188 claims used for BM25 and Sentence-BERT."""

import json
import time
from pathlib import Path

from src.retrieval.hybrid import HybridRetriever


CLAIMS_PATH = Path("data/processed/claims_dev.jsonl")
OUTPUT_PATH = Path("reports/hybrid_retrieval_metrics.json")


def load_claims():
    with CLAIMS_PATH.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main():
    retriever = HybridRetriever()

    recalls_5 = []
    recalls_10 = []
    reciprocal_ranks = []

    start = time.perf_counter()

    try:
        for claim in load_claims():
            gold_docs = set(
                str(doc_id)
                for doc_id in claim.get("evidence", {})
            )

            if not gold_docs:
                continue

            results = retriever.search(claim["claim"])

            ranked_docs = [
                str(result["doc_id"])
                for result in results
            ]

            recalls_5.append(
                len(gold_docs.intersection(ranked_docs[:5]))
                / len(gold_docs)
            )

            recalls_10.append(
                len(gold_docs.intersection(ranked_docs[:10]))
                / len(gold_docs)
            )

            rr = 0.0

            for rank, doc_id in enumerate(ranked_docs, start=1):
                if doc_id in gold_docs:
                    rr = 1.0 / rank
                    break

            reciprocal_ranks.append(rr)

            evaluated = len(recalls_5)

            if evaluated % 25 == 0:
                print(f"Evaluated {evaluated} claims")

    finally:
        retriever.close()

    n = len(recalls_5)

    metrics = {
        "evaluated_claims": n,
        "recall_at_5": sum(recalls_5) / n,
        "recall_at_10": sum(recalls_10) / n,
        "mrr_at_10": sum(reciprocal_ranks) / n,
        "total_seconds": round(
            time.perf_counter() - start, 2
        )
    }

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    print("\nHYBRID RETRIEVAL EVALUATION")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()