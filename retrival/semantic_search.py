from embeddings.semantic_embedder import embed_query
from vector_store.qdrant_store import get_client, COLLECTION_NAME
from retrival.dense_search import dense_search,fetch_full_text


def semantic_search(query:str,top_k:int=10):
    client=get_client()
    query_vector=embed_query(query)
    results=client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="semantic",
        limit=top_k*4,
    )
    seen_items={}
    for p in results.points:
        item_id=p.payload["item_id"]
        if item_id not in seen_items:
            seen_items[item_id]={
                "item_id":item_id,
                "score":p.score,
                "image_url":p.payload.get("main_image_url") or p.payload.get("image_url"),
                "matched_via":"semantic"
            }
            
        if len(seen_items)>=top_k:
            break
    for item_id,product in seen_items.items():
        product["full_text"]=fetch_full_text(item_id,client=client)
    return list(seen_items.values())
