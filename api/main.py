import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from api.upload import router as upload_router
from api.health import router as health_router
from api.ask import router as ask_router

app = FastAPI(title="ABO-RAG")
app.include_router(health_router, prefix="/api")
app.include_router(upload_router, prefix="/api")
app.include_router(ask_router, prefix="/api")