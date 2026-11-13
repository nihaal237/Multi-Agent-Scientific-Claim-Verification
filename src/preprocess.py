"""prepare the scientific papers so our agents can search them."""

import json
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def save_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def preprocess():
    corpus = load_jsonl(RAW / "corpus.jsonl")

    documents = []
    chunks = []

    for paper in corpus:
        doc_id = str(paper["doc_id"])
        title = paper["title"].strip()
        sentences = paper["abstract"]

        documents.append({
            "doc_id": doc_id,
            "title": title,
            "abstract": " ".join(sentences),
            "sentences": sentences
        })

        #creating chunks-> Suppose an abstract contains five sentences, and only sentence 4 contradicts a user's claim.

        #If we store only the entire abstract, retrieval may find the correct paper but not the specific evidence.

        #By storing individual sentences as chunks, we can retrieve precise evidence and evaluate it using SciFact's ground-truth sentence indices.

        # One title+abstract sentence per chunk
        for sentence_id, sentence in enumerate(sentences):
            sentence = sentence.strip()

            if not sentence:
                continue

            chunks.append({
                "chunk_id": f"{doc_id}_{sentence_id}",
                "doc_id": doc_id,
                "sentence_id": sentence_id,
                "title": title,
                "text": sentence,
                "embedding_text": f"{title}. {sentence}"
            })

    save_jsonl(OUT / "documents.jsonl", documents)
    save_jsonl(OUT / "chunks.jsonl", chunks)

    # Preserve original claim annotations for evaluation
    for split in ("train", "dev", "test"):
        claims = load_jsonl(RAW / f"claims_{split}.jsonl")
        save_jsonl(OUT / f"claims_{split}.jsonl", claims)

    print("Documents:", len(documents))
    print("Sentence chunks:", len(chunks))

    print("Processed data saved to:", OUT)


if __name__ == "__main__":
    preprocess()
