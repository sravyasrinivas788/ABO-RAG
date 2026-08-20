import sys
import json
from pathlib import Path
from embeddings.chunker import chunk_for_clip
from embeddings.text_embedder import embed_texts_batch
from embeddings.image_embedder import embed_images_batch
from vector_store.qdrant_store import get_client, create_collection, build_text_point, build_image_point, upsert_points

DATA_PATH = Path(__file__).parent.parent / "data" / "products_clean.jsonl"
PROGRESS_PATH = Path(__file__).parent.parent / "data" / "embed_progress.json"
BATCH_SIZE=32
INCLUDE_OTHER_IMAGES=False

def load_all_products():
    products = []
    with open(DATA_PATH,encoding="utf-8") as f:
        for line in f:
            products.append(json.loads(line))
    return products

def load_progress() -> int:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))["next_batch_start"]
    return 0

def save_progress(next_batch_start: int):
    PROGRESS_PATH.write_text(json.dumps({"next_batch_start": next_batch_start}), encoding="utf-8")

def clear_progress():
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()

def process_batch(client,products_batch):
    text_points=[]
    image_points=[]
    chunk_records=[]
    for p in products_batch:
        for i, chunk_text in enumerate(chunk_for_clip(p)):
            chunk_records.append((p,i,chunk_text))
    if chunk_records:
        chunk_vectors=embed_texts_batch([r[2] for r in chunk_records])
        for (p,i,chunk_text),vector in zip(chunk_records,chunk_vectors):
            text_points.append(build_text_point(p,i,chunk_text,vector))
    image_records=[]
    for p in products_batch:
        main_image_url=p.get("main_image_url")
        if main_image_url:
            image_records.append((p,"main",0,main_image_url))
        if INCLUDE_OTHER_IMAGES:
            for i, url in enumerate(p.get("other_image_urls",[])):
                image_records.append((p,"other",i,url))
    if image_records:
        image_vectors=embed_images_batch([r[3] for r in image_records])
        for (p,category,index,url),vector in zip(image_records,image_vectors):
            if vector is None:
                print(f"Skipping image point for {p.get('item_id')} ({category}/{index}): failed to embed {url}")
                continue
            image_points.append(build_image_point(p,category,index,url,vector))
    
    all_points=text_points+image_points
    if all_points:
        upsert_points(client,all_points)
    return len(text_points), len(image_points)

def run_full_batch(resume: bool=False):
    client=get_client()
    start = load_progress() if resume else 0
    if not resume:
        clear_progress()
    create_collection(client,recreate=(start==0))
    products=load_all_products()
    total_text_points=0
    total_image_points=0
    for i in range(start,len(products),BATCH_SIZE):
        batch=products[i:i+BATCH_SIZE]
        text_count,image_count=process_batch(client,batch)
        total_text_points+=text_count
        total_image_points+=image_count
        save_progress(i+BATCH_SIZE)
        print(f"Processed batch {i//BATCH_SIZE+1}: {text_count} text points, {image_count} image points")
    clear_progress()




