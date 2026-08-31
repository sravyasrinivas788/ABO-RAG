import sys
import io
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Form
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrival.dense_search import embed_text_query,embed_image_query,dense_search,combine_query_vectors
from retrival.llm_generate import generate_answer
from retrival.hybrid_search import hybrid_search_from_vector,hybrid_search

router = APIRouter()

@router.post("/ask")
async def ask_question(question:str=Form(None),image: UploadFile=None):
    if not question and not image:
        return {"error": "Please provide either a question or an image."}
    if question and image is not None:
        search_path="text+image"
        image_bytes=await image.read()
        image_pil=Image.open(io.BytesIO(image_bytes)).convert("RGB")
        query_vector=combine_query_vectors(embed_text_query(question),embed_image_query(image_pil))
        retrived=hybrid_search_from_vector(query_vector,query_text=question,top_k=5)
    elif image is not None:
        search_path="image"
        image_pil=Image.open(io.BytesIO(await image.read())).convert("RGB")
        query_vector=embed_image_query(image_pil)
        retrived=hybrid_search_from_vector(query_vector,query_text=None,top_k=5)
    else:
        search_path="text"
        retrived=hybrid_search(question,top_k=5)

    default_question = "The user uploaded an image and the products below were retrieved as visually similar matches. Describe these matching products."
    result= generate_answer(question or default_question,retrived)
    result["debug"] = {
        "search_path": search_path,
        "retrieved_chunks": [
            {
                "item_id": r["item_id"],
                "matched_via": r["matched_via"],
                "score": round(r["score"], 5),
                "text_preview": r["full_text"][:150],
            }
            for r in retrived
        ],
    }
    return result


    




   