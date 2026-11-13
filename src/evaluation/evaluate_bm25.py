"""evaluate BM25 using the same 188 development claims."""

import json
import time
from pathlib import Path

from src.retrieval.bm25 import BM25Retriever


CLAIMS_PATH = Path("data/processed/claims_dev.jsonl")
OUTPUT_PATH = Path("reports/bm25_retrieval_metrics.json")


def load_claims():
    with CLAIMS_PATH.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main():
    print("Loading BM25 retriever...")

    retriever = BM25Retriever()
    claims = load_claims()

    recall_5 = []
    recall_10 = []
    reciprocal_ranks = []

    start_time = time.time()

    evaluated = 0

    for claim in claims:
        gold_docs = set(
            str(doc_id)
            for doc_id in claim.get("evidence", {})
        )

        if not gold_docs:
            continue

        results = retriever.search(
            claim["claim"],
            top_k=10
        )

        ranked_docs = [
            str(result["doc_id"])
            for result in results
        ]

        recall_5.append(
            len(gold_docs.intersection(ranked_docs[:5]))
            / len(gold_docs)
        )

        recall_10.append(
            len(gold_docs.intersection(ranked_docs[:10]))
            / len(gold_docs)
        )

        rr = 0.0

        for rank, doc_id in enumerate(ranked_docs, start=1):
            if doc_id in gold_docs:
                rr = 1.0 / rank
                break

        reciprocal_ranks.append(rr)

        evaluated += 1

        if evaluated % 25 == 0:
            print(f"Evaluated {evaluated} claims")

    metrics = {
        "evaluated_claims": evaluated,
        "recall_at_5": sum(recall_5) / evaluated,
        "recall_at_10": sum(recall_10) / evaluated,
        "mrr_at_10": sum(reciprocal_ranks) / evaluated,
        "total_seconds": round(
            time.time() - start_time, 2
        )
    }

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    print("\nBM25 RETRIEVAL EVALUATION")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()