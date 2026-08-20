import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance,VectorParams,PointStruct

COLLECTION_NAME="abo_products"
VECTOR_SIZE=512

def get_client()->QdrantClient:
    return QdrantClient(url="http://localhost:6333")

def create_collection(client: QdrantClient, recreate: bool=False):
    if recreate and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

def make_point_id(item_id: str, modality: str, index: int) -> str:
    """Deterministic UUID from item_id+modality+index -- re-running the same
    upsert twice UPDATES the same point instead of creating duplicates."""
    key = f"{item_id}:{modality}:{index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

def build_text_point(product:dict,chunk_index:int,chunk_text:str,vector: list[float])->PointStruct:
    return PointStruct(
        id=make_point_id(product["item_id"],"text",chunk_index),
        vector=vector,
        payload={
            "item_id": product["item_id"],
            "modality": "text",
            "chunk_index": chunk_index,
            "text": chunk_text,
            "category": product.get("category_primary"),
            "product_type": product.get("product_type"),
            "brand": product.get("brand"),
            "main_image_url": product.get("main_image_url"),

        },


    )

def build_image_point(product:dict,image_role:str,index:int,image_url:str,vector:list[float])->PointStruct:
    return PointStruct(

        id=make_point_id(product["item_id"],"image",index),
        vector=vector,
         payload={
            "item_id": product["item_id"],
            "modality": "image",
            "image_role": image_role,   # "main" or "other"
            "image_url": image_url,
            "category": product.get("category_primary"),
            "product_type": product.get("product_type"),
            "brand": product.get("brand"),
        },

    )


def upsert_points(client: QdrantClient, points: list[PointStruct]):
    client.upsert(collection_name=COLLECTION_NAME, points=points)
