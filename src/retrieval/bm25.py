import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


DOCUMENTS_PATH = Path("data/processed/documents.jsonl")


def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def load_documents():
    with DOCUMENTS_PATH.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


class BM25Retriever:
    def __init__(self):
        self.documents = load_documents()

        # Create tokens for each paper
        self.tokenized_corpus = [
            tokenize(
                doc["title"] + " " + doc["abstract"]
            )
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=10):
        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [
            {
                "doc_id": self.documents[i]["doc_id"],
                "title": self.documents[i]["title"],
                "score": float(scores[i])
            }
            for i in ranked_indices
        ]


if __name__ == "__main__":
    retriever = BM25Retriever()

    query = "Vitamin D supplementation reduces depression."

    results = retriever.search(query, top_k=5)

    print("\nCLAIM:", query)
    print("\nTOP 5 BM25 PAPERS:\n")

    for rank, doc in enumerate(results, start=1):
        print(f"{rank}. {doc['title']}")
        print("Paper ID:", doc["doc_id"])
        print("BM25 score:", round(doc["score"], 4))
        print()