# ABO-RAG

Ingestion pipeline that normalizes Amazon Berkeley Objects (ABO) product
listings into a clean schema suitable for embedding and retrieval-augmented
generation (RAG).

## Project structure

```
data/
  listings_0.json        JSONL file of raw ABO product listings
ingestion/
  loader.py               Streams listings from the JSONL file
  normalize_text.py        Picks a preferred-language value from localized text fields
  normalize_fields.py      Extracts single-value fields and category/node info
  normalize_measurement.py Extracts normalized dimensions and weight
  image_urls.py            Builds Amazon CDN image URLs from image ids
  product_schema.py        Combines the above into a single Product record
```

## Setup

```
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Inspect the raw data:

```
python -m ingestion.loader
```

Build a normalized `Product` from a sample listing and run the built-in checks:

```
python -m ingestion.product_schema
```

Each module in `ingestion/` also runs as a standalone script when executed
this way (as `-m`, from the project root), for quick spot-checks against real
listing data.

## Running the API

```
uvicorn api.main:app --reload
```

Run this from the project root (not from inside `api/`).

## Running everything with Docker

```
cp .env.example .env   # fill in GROQ_API_KEY
docker compose up 
```

This starts three services:

- `qdrant` -- vector store, on `localhost:6333`
- `postgres` -- conversation history store, on `localhost:5432`
- `app` -- the FastAPI service (this repo, built from the `Dockerfile`), on `localhost:8000`

The `app` container talks to the other two over the compose network
(`http://qdrant:6333`, `postgres` host), regardless of what `POSTGRES_URL`
in your `.env` is set to for host-based runs -- `docker-compose.yml`
overrides both to the in-network service names.

On a brand new stack, Qdrant starts empty. Populate it once with:

```
curl -X POST http://localhost:8000/api/upload
```

which materializes `data/listings_0.json` into `data/products_clean.jsonl`
and embeds everything into Qdrant in the background. Check progress with
`curl http://localhost:8000/api/status`.

## Using the /ask endpoint

`POST /api/ask` is a `multipart/form-data` request with these fields:

- `session_id` (required) -- any string identifying the conversation. Pass
  the same value on follow-up questions to get pronoun/context resolution
  ("what colors does it come in") against the earlier turns; a new value
  starts a fresh conversation with no history.
- `question` (optional) -- the text question.
- `image` (optional) -- an image file to search with.

At least one of `question` or `image` must be provided.

Text-only search:

```
curl -X POST http://localhost:8000/api/ask \
  -F "session_id=demo-session-1" \
  -F "question=mobile phone cover"
```

Follow-up in the same session (reuse the same `session_id`):

```
curl -X POST http://localhost:8000/api/ask \
  -F "session_id=demo-session-1" \
  -F "question=what colors does it come in"
```

Image-only search (find visually similar products):

```
curl -X POST http://localhost:8000/api/ask \
  -F "session_id=demo-session-1" \
  -F "image=@/path/to/photo.jpg"
```

Combined text + image search:

```
curl -X POST http://localhost:8000/api/ask \
  -F "session_id=demo-session-1" \
  -F "question=something similar but in leather" \
  -F "image=@/path/to/photo.jpg"
```

Response shape:

```json
{
  "answer": "...",
  "citations": [{ "item_id": "B0853X2F4M", "image_url": "https://..." }],
  "debug": {
    "search_path": "text",
    "original_question": "...",
    "standardized_question": "...",
    "history_used": ["..."],
    "retrieved_chunks": [ { "item_id": "...", "matched_via": "...", "score": 0.0, "text_preview": "..." } ]
  }
}
``` 