import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from embeddings.clip_model import get_model, get_processor
import torch
from vector_store.qdrant_store import get_client, COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, Range,MatchValue

def embed_text_query(text:str)->list[float]:
    model=get_model()
    processor=get_processor()
    inputs=processor(text=[text], return_tensors="pt", padding=True,truncation=True)
    with torch.no_grad():
        features=model.get_text_features(**inputs).pooler_output
    vector=features[0]
    vector=vector/vector.norm()
    return vector.numpy().tolist()

def embed_image_query(image)->list[float] | None:
    model=get_model()
    processor=get_processor()
    inputs=processor(images=[image], return_tensors="pt")
    with torch.no_grad():
        features=model.get_image_features(**inputs).pooler_output
    vector=features[0]
    vector=vector/vector.norm()
    return vector.numpy().tolist()


def dense_search(query_vector:list[float], top_k:int=5):
    client=get_client()
    results=client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k*4,
    )
    seen_items = {}
    for p in results.points:
        item_id = p.payload["item_id"]
        if item_id not in seen_items:
            seen_items[item_id] = {
                "item_id": item_id,
                "score": p.score,
                "image_url": p.payload.get("main_image_url") or p.payload.get("image_url"),
                "matched_via": p.payload["modality"],  
            }
        if len(seen_items) >= top_k:
            break
    for item_id, product in seen_items.items():
        chunk_results, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[
                FieldCondition(key="item_id", match=MatchValue(value=item_id)),
                FieldCondition(key="modality", match=MatchValue(value="text")),
            ]),
            limit=20,
        )
        chunks_sorted = sorted(chunk_results, key=lambda c: c.payload["chunk_index"])
        product["full_text"] = " | ".join(c.payload["text"] for c in chunks_sorted)

    return list(seen_items.values())


    
        

        

   