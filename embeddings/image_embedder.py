import io 
import requests
import torch
from PIL import Image
from embeddings.clip_model import get_model,get_processor

def fetch_image(url: str, timeout: int=10)->Image.Image | None:
    try:
        response=requests.get(url,timeout=timeout)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except:
        print(f"Failed to fetch{url}")
        return None


def embed_image(image_url:str)->list[float] | None:
    image=fetch_image(image_url)
    if image is None:
        return None
    model=get_model()
    processor=get_processor()
    inputs=processor(images=[image], return_tensors="pt")
    with torch.no_grad():
        features=model.get_image_features(**inputs).pooler_output
    vector=features[0]
    vector=vector/vector.norm()
    return vector.numpy().tolist()

def embed_images_batch(image_urls: list[str]) -> list[list[float] | None]:
    """Fetches images one at a time (network calls can't be batched), but
    embeds all successfully-fetched images in a single model call."""
    images = []
    valid_indices = []
    for i, url in enumerate(image_urls):
        img = fetch_image(url)
        if img is not None:
            images.append(img)
            valid_indices.append(i)

    results: list[list[float] | None] = [None] * len(image_urls)
    if not images:
        return results

    model = get_model()
    processor = get_processor()
    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs).pooler_output

    normed = features / features.norm(dim=1, keepdim=True)
    vectors = normed.numpy().tolist()

    for idx, vec in zip(valid_indices, vectors):
        results[idx] = vec

    return results


    
if __name__ == "__main__":
    # real image URL, from the actual drawer-slide product
    test_url = "https://m.media-amazon.com/images/I/619y9YG9cnL.jpg"
    vector = embed_image(test_url)
    if vector is None:
        print("FAILED -- check your network can reach m.media-amazon.com")
    else:
        print(f"Success. Vector length: {len(vector)}")
        assert len(vector) == 512
