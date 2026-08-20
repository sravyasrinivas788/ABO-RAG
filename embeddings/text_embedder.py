import torch
from embeddings.clip_model import get_model, get_processor

def embed_text(text:str)->list[float]:
    model=get_model()
    processor=get_processor()
    inputs=processor(text=[text], return_tensors="pt", padding=True,truncation=True)
    with torch.no_grad():
        features=model.get_text_features(**inputs).pooler_output
    vector=features[0]
    vector=vector/vector.norm()
    return vector.numpy().tolist()

    
def embed_texts_batch(texts: list[str]) -> list[list[float]]:
    model = get_model()
    processor = get_processor()
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs).pooler_output
    normed = features / features.norm(dim=1, keepdim=True)
    return normed.numpy().tolist()