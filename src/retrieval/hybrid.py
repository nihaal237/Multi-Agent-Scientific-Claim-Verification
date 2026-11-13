"""first run will download the cross-encoder model.

You should see 10 ranked research papers.

Note: A cross-encoder score is a relevance score, not a probability that the paper supports the claim."""

import json
from pathlib import Path

import yaml
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.retrieval.bm25 import BM25Retriever


class HybridRetriever:
    def __init__(self):
        with open("configs/retrieval.yaml", encoding="utf-8") as f:
            self.retrieval_config = yaml.safe_load(f)

        with open("configs/hybrid.yaml", encoding="utf-8") as f:
            self.hybrid_config = yaml.safe_load(f)

        self.bm25 = BM25Retriever()

        self.embedding_model = SentenceTransformer(
            self.retrieval_config["embedding_model"]
        )

        self.reranker = CrossEncoder(
            self.hybrid_config["cross_encoder_model"]
        )

        self.qdrant = QdrantClient(
            path="artifacts/index/qdrant"
        )

        self.documents = {
            str(doc["doc_id"]): doc
            for doc in self.bm25.documents
        }

    def search(self, claim):
        config = self.hybrid_config

        # 1. BM25 candidate papers
        bm25_results = self.bm25.search(
            claim,
            top_k=config["bm25_top_k"]
        )

        # 2. Semantic candidate sentences
        query_vector = self.embedding_model.encode(
            claim,
            normalize_embeddings=True
        ).tolist()

        semantic_hits = self.qdrant.query_points(
            collection_name=self.retrieval_config["collection_name"],
            query=query_vector,
            limit=config["semantic_chunk_fetch"],
            with_payload=True
        ).points

        # 3. Aggregate sentence hits into unique papers
        semantic_ids = []
        seen = set()

        for hit in semantic_hits:
            doc_id = str(hit.payload["doc_id"])

            if doc_id not in seen:
                semantic_ids.append(doc_id)
                seen.add(doc_id)

            if len(semantic_ids) >= config["semantic_top_k"]:
                break

        # 4. Combine candidate papers
        candidate_ids = set()

        for result in bm25_results:
            candidate_ids.add(str(result["doc_id"]))

        candidate_ids.update(semantic_ids)

        candidates = [
            self.documents[doc_id]
            for doc_id in sorted(candidate_ids)
            if doc_id in self.documents
        ]

        # 5. Cross-encoder reranking
        pairs = [
            (claim, doc["title"] + ". " + doc["abstract"])
            for doc in candidates
        ]

        scores = self.reranker.predict(
            pairs,
            batch_size=16
        )

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True
        )

        return [
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "score": float(score)
            }
            for doc, score in ranked[:config["rerank_top_k"]]
        ]

    def close(self):
        self.qdrant.close()


if __name__ == "__main__":
    retriever = HybridRetriever()

    try:
        claim = "Vitamin D supplementation reduces depression."

        results = retriever.search(claim)

        print("\nCLAIM:", claim)

        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. {result['title']} "
                f"(score={result['score']:.4f})"
            )
    finally:
        retriever.close()