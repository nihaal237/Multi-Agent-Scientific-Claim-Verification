"""Load SciFact's development claims, Search Qdrant for each claim, Retrieve the top 10 distinct papers, Compare their IDs against the annotated evidence paper IDs.
   Calculate Recall@5, Recall@10, and MRR@10, Save the results to reports/semantic_retrieval_metrics.json."""

import json
import time
from pathlib import Path

import yaml
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

CONFIG_PATH = Path("configs/retrieval.yaml")
CLAIMS_PATH = Path("data/processed/claims_dev.jsonl")
QDRANT_PATH = "artifacts/index/qdrant"

TOP_K = 10


def load_claims():
    with CLAIMS_PATH.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main():
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = SentenceTransformer(config["embedding_model"])

    client = QdrantClient(path=QDRANT_PATH)

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

        query_embedding = model.encode(
            claim["claim"],
            normalize_embeddings=True
        ).tolist()

        # Retrieve more sentences to obtain distinct papers
        results = client.query_points(
            collection_name=config["collection_name"],
            query=query_embedding,
            limit=100,
            with_payload=True
        ).points

        ranked_docs = []
        seen = set()

        for hit in results:
            doc_id = str(hit.payload["doc_id"])

            if doc_id not in seen:
                ranked_docs.append(doc_id)
                seen.add(doc_id)

            if len(ranked_docs) >= TOP_K:
                break

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
        "total_seconds": round(time.time() - start_time, 2)
    }

    Path("reports").mkdir(exist_ok=True)

    Path("reports/semantic_retrieval_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    print("\nRETRIEVAL EVALUATION")
    print(json.dumps(metrics, indent=2))

    client.close()


if __name__ == "__main__":
    main()
