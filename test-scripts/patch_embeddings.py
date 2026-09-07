"""
One-time patch: adds height_in/width_in/weight_lb to EXISTING Qdrant
points. Payload-only change -- does NOT touch vectors, so no re-embedding
needed.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client.models import Filter, FieldCondition, MatchValue
from vector_store.qdrant_store import get_client, COLLECTION_NAME

DATA_PATH = Path(__file__).parent.parent / "data" / "products_clean.jsonl"

client = get_client()
count = 0

with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        p = json.loads(line)
        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={
                "height_in": p.get("height_in"),
                "width_in": p.get("width_in"),
                "length_in": p.get("length_in"),
                "weight_lb": p.get("weight_lb"),
            },
            points=Filter(must=[FieldCondition(key="item_id", match=MatchValue(value=p["item_id"]))]),
        )
        count += 1

print(f"Patched dimension/weight payload on {count} products.")