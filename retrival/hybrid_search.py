from retrival.dense_search import dense_search,fetch_full_text,embed_text_query
from retrival.bm25_search import bm25_search
from retrival.semantic_search import semantic_search
import logging
from retrival.reranker import rerank
logger=logging.getLogger("abo_rag_logger")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

RRF_K=60

def reciprocal_rank_fusion(dense_results:list[dict],bm25_results:list[dict],top_k: int=5)->list[dict]:
    scores,sources,info={},{},{}
    for rank,r in enumerate(dense_results,start=1):
        scores[r["item_id"]]=scores.get(r["item_id"],0)+1/(RRF_K+rank)
        sources.setdefault(r["item_id"], set()).add(r.get("matched_via", "dense"))
        info.setdefault(r["item_id"],r)

    for rank,r in enumerate(bm25_results,start=1):
        scores[r["item_id"]]=scores.get(r["item_id"],0)+1/(RRF_K+rank)
        sources.setdefault(r["item_id"],set()).add(r.get("matched_via", "bm25"))
        info.setdefault(r["item_id"],r)

    ranked_items=sorted(scores.items(),key=lambda x:x[1],reverse=True)[:top_k]
    results=[]
    for item_id,score in ranked_items:
        matched=sources[item_id]
        full_text=info[item_id].get("full_text")
        if full_text is None:
            full_text=fetch_full_text(item_id)
        results.append({
            "item_id":item_id,
            "score":score,
            "image_url":info[item_id].get("image_url"),
            "matched_via":",".join(sorted(matched)),
            "full_text":full_text
        
        })
    logger.info(f"RRF results: {[r['item_id'] for r in results]}")
    for r in results:
        logger.info(f"  {r['item_id']}: score={r['score']:.4f}, matched_via={r['matched_via']}")
    return results

def hybrid_search_from_vector(query_vector:list[float],query_text:str,top_k:int=5)->list[dict]:
    logger.info(f"routing for hybrid search with query vector and text using CLIP: {query_text}")
    dense_results=dense_search(query_vector,top_k=top_k*3)
    logger.info(f"Dense search results: {[r['item_id']+ ': ' + str(r['score']) for r in dense_results]}")
    bm25_results=bm25_search(query_text,top_k=top_k*3) if query_text else []
    logger.info(f"BM25 search results: {[r['item_id'] + ': ' + str(r['score']) for r in bm25_results]}")
    fused=reciprocal_rank_fusion(dense_results,bm25_results,top_k=top_k*3)
    reranked=rerank(query_text,fused,top_k=top_k) if query_text else fused[:top_k]
    return reranked

def hybrid_search(query:str,top_k:int=5)->list[dict]:
    logger.info(f"routing for semantic BGE search {query}")
    semantic_results=semantic_search(query,top_k=top_k*3)
    logger.info(f"Semantic search results: {[r['item_id']+ ': ' + str(r['score']) for r in semantic_results]}")
    bm25_results=bm25_search(query,top_k=top_k*3) if query else []
    logger.info(f"BM25 search results: {[r['item_id'] + ': ' + str(r['score']) for r in bm25_results]}")
    fused=reciprocal_rank_fusion(semantic_results,bm25_results,top_k=top_k*3)
    reranked=rerank(query,fused,top_k=top_k) if query else fused[:top_k]
    return reranked
   