import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.chunker import chunk_for_clip
from embeddings.text_embedder import embed_texts_batch

DATA_PATH = Path(__file__).parent.parent / "data" / "products_clean.jsonl"


def load_trial_products(n=10):
    products = []
    with open(DATA_PATH) as f:
        for line in f:
            products.append(json.loads(line))
            if len(products) >= n:
                break
    return products


def cosine_sim(a, b):
    return float(np.dot(np.array(a), np.array(b)))


if __name__ == "__main__":
    products = load_trial_products(10)

    
    records = []
    for p in products:
        chunks = chunk_for_clip(p)
        for i, chunk_text in enumerate(chunks):
            records.append({"item_id": p["item_id"], "chunk_index": i, "text": chunk_text})

    print(f"{len(products)} products -> {len(records)} chunks")

    vectors = embed_texts_batch([r["text"] for r in records])
    for r, v in zip(records, vectors):
        r["vector"] = v

    print("\nTop 5 most similar chunk pairs:")
    pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            sim = cosine_sim(records[i]["vector"], records[j]["vector"])
            pairs.append((sim, f"{records[i]['item_id']}#{records[i]['chunk_index']}", f"{records[j]['item_id']}#{records[j]['chunk_index']}"))

    pairs.sort(reverse=True)
    for sim, a, b in pairs[:5]:
        print(f"  {sim:.3f}  |  {a}  <->  {b}")