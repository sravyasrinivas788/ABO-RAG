import sys
import io
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Form
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrival.dense_search import embed_text_query,embed_image_query,dense_search
from retrival.llm_generate import generate_answer

router = APIRouter()

@router.post("/ask")
async def ask_question(question:str=Form(None),image: UploadFile=None):
    if not question and not image:
        return {"error": "Please provide either a question or an image."}
    if image is not None:
        image_bytes=await image.read()
        image_pil=Image.open(io.BytesIO(image_bytes)).convert("RGB")
        query_vector=embed_image_query(image_pil)
    else:
        query_vector=embed_text_query(question)
    retrived=dense_search(query_vector,top_k=5)
    result=generate_answer(question or "Describe the image:",retrived)
    return result





   