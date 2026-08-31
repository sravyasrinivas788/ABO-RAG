from sentence_transformers import SentenceTransformer
MODEL_NAME =  "BAAI/bge-base-en-v1.5"

QUERY_INSTRUCTION= "Please generate a vector representation of the following query for semantic search. The vector should capture the meaning and intent of the query, enabling effective retrieval of relevant documents from a database or corpus. Ensure that the vector is normalized and suitable for use in similarity comparisons."

_model=None

def get_semantic_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_query(text:str)->list[float]:
    model=get_semantic_model()
    vector=model.encode(QUERY_INSTRUCTION + text,normalize_embeddings=True)
    return vector.tolist()

def embed_passage(text:str)->list[float]:
    model=get_semantic_model()
    vector=model.encode(text,normalize_embeddings=True)
    return vector.tolist()

def embed_passages_batch(texts: list[str]) -> list[list[float]]:
    model = get_semantic_model()
    vectors = model.encode(texts, normalize_embeddings=True,batch_size=32)
    return vectors.tolist()