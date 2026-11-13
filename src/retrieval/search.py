"""get five sentences ranked by semantic similarity. They may or may not support the claim—that judgment comes later from the Evidence and Verification Agents."""

import yaml
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

with open("configs/retrieval.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model = SentenceTransformer(config["embedding_model"])

client = QdrantClient(path="artifacts/index/qdrant")

claim = "Vitamin D supplementation reduces depression."

query_vector = model.encode(
    claim,
    normalize_embeddings=True
).tolist()

results = client.query_points(
    collection_name=config["collection_name"],
    query=query_vector,
    limit=5,
    with_payload=True
).points

print("\nCLAIM:", claim)
print("\nTOP 5 RETRIEVED SENTENCES:\n")

for rank, hit in enumerate(results, start=1):
    p = hit.payload

    print(f"Rank {rank}")
    print("Paper ID:", p["doc_id"])
    print("Title:", p["title"])
    print("Sentence:", p["text"])
    print("Similarity:", round(hit.score, 4))
    print("-" * 60)

client.close()
