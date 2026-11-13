"""It reads every sentence chunk, generates an embedding using Sentence-BERT, and saves the vector alongside the original sentence, paper ID, and sentence index in Qdrant."""

import json
import shutil
from pathlib import Path

import yaml
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
CONFIG_PATH = Path("configs/retrieval.yaml")
INDEX_PATH = Path("artifacts/index/qdrant")


def load_chunks():
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def main():
    with CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} sentence chunks")

    model_name = config["embedding_model"]
    collection = config["collection_name"]
    batch_size = int(config["batch_size"])
    upload_size = int(config["upload_batch_size"])

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Create a fresh index for reproducibility
    if INDEX_PATH.exists():
        shutil.rmtree(INDEX_PATH)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(INDEX_PATH))

    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=model.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE
        )
    )

    total = len(chunks)

    for start in range(0, total, upload_size):
        batch = chunks[start:start + upload_size]

        texts = [chunk["embedding_text"] for chunk in batch]

        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        points = []

        for i, (chunk, vector) in enumerate(
            zip(batch, vectors)
        ):
            points.append(
                models.PointStruct(
                    id=start + i,
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "sentence_id": chunk["sentence_id"],
                        "title": chunk["title"],
                        "text": chunk["text"]
                    }
                )
            )

        client.upload_points(
            collection_name=collection,
            points=points,
            batch_size=upload_size
        )

        print(
            f"Indexed {min(start + len(batch), total)}/{total}"
        )

    info = client.get_collection(collection)
    print("Total indexed points:", info.points_count)

    client.close()
    print("Qdrant index saved at:", INDEX_PATH)


if __name__ == "__main__":
    main()
