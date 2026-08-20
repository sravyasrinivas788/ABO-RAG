import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import APIRouter, UploadFile, File,BackgroundTasks

from ingestion.materialize import materialize
from embeddings.batch_embed import run_full_batch

router = APIRouter()
_status = {"state": "idle", "detail": ""}

def run_ingestion_pipeline(resume: bool = False):
    global _status
    try:
        if not resume:
            _status = {"state": "materializing", "detail": "Materializing products..."}
            materialize()
        _status = {"state": "embedding", "detail": "Embedding products..." if not resume else "Resuming embedding from last completed batch..."}
        run_full_batch(resume=resume)
        _status = {"state": "completed", "detail": "Ingestion pipeline completed successfully."}
    except Exception as e:
        _status = {"state": "error", "detail": str(e)}


@router.post("/upload")
def upload(background_tasks: BackgroundTasks, resume: bool = False):
    background_tasks.add_task(run_ingestion_pipeline, resume=resume)
    return {"message": f"Ingestion pipeline started in the background{' (resuming from last completed batch)' if resume else ''}."}

@router.get("/status")
def upload_status():
    return _status

