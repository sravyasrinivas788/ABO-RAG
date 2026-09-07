from sentence_transformers import CrossEncoder

_reranker=None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def rerank(query:str, candidates: list[dict],top_k:int=5)->list[dict]:
    if not candidates:
        return candidates
    pairs=[(query,c["full_text"]) for c in candidates]
    scores=get_reranker().predict(pairs)
    for c,score in zip(candidates,scores):
        c["rerank_score"]=float(score)
    return sorted(candidates,key=lambda x:x["rerank_score"],reverse=True)[:top_k]

