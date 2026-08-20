from transformers import CLIPModel , CLIPProcessor
from embeddings.config import MODEL_NAME, DEVICE

_model=None
_processor=None

def get_model():
    global _model
    if _model is None:
        _model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        _model.eval()
    return _model
    
def get_processor():
    global _processor
    if _processor is None:
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return _processor