from fastapi import APIRouter, UploadFile, File,BackgroundTasks
router=APIRouter()

@router.get("/health")
def health_check():
    return {"status": "healthy"}