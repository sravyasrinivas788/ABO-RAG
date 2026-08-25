import json
import sys
from pathlib import Path
from rank_bm25 import BM25Okapi
sys.path.insert(0, str(Path(__file__).parent.parent))
from embeddings.chunker import chunk_for_clip

DATA_PATH = Path(__file__).parent.parent / "data" / "products_clean.jsonl"

_bm25 = None
_records = None

def build_index():
    global _bm25,_records
    records=[]
    with open(DATA_PATH,encoding="utf-8") as f:
        for line in f:
            p=json.loads(line)
            for i,chunk_text in enumerate(chunk_for_clip(p)):
                records.append({"item_id":p["item_id"],"chunk_index":i,"text":chunk_text,"image_url":p.get("main_image_url") or p.get("image_url")})
    _records=records
    _bm25=BM25Okapi([r["text"].lower().split() for r in records])

def bm25_search(query:str,top_k:int=10)->list[dict]:
    if _bm25 is None:
        build_index()
    scores=_bm25.get_scores(query.lower().split())
    ranked_index=sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)
    seen_iteems={}
    for idx in ranked_index:
        item_id=_records[idx]["item_id"]
        if item_id not in seen_iteems:
            seen_iteems[item_id]={
                "item_id":item_id,
                "score":scores[idx],
                "image_url":_records[idx]["image_url"],
                "matched_via":"text",
            }
        if len(seen_iteems)>=top_k:
            break
    return list(seen_iteems.values())