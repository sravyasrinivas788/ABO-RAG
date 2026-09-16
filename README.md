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

## Retrieval evaluation

`eval/` holds a retrieval-only test harness -- it exercises `hybrid_search()`
directly (no LLM generation involved) against a golden set built from the
catalog itself, so ground truth is known by construction. Run everything
from the project root, with Qdrant populated (see "Running everything with
Docker" above) and `GROQ_API_KEY` set.

### 1. Build the golden set

```
python -m eval.build_golden_set
```

Samples up to 8 products per `product_type` (capped to the 30 most frequent
categories) and, for each, generates:

- an item-level natural-language query with one distinguishing detail
  (e.g. `"blue phone case with butterfly design"`), written to
  `eval/retrieval_golden.jsonl` with the source `item_id` as ground truth
- one vague, category-level browsing query per category (e.g.
  `"pillow options"`), written to `eval/retrieval_golden_vague.jsonl` --
  these have no single correct item, so they're scored differently (see below)

Review `retrieval_golden.jsonl` afterwards and discard/fix any query that
isn't uniquely answered by its `item_id`.

### 2. Run the eval

```
python -m eval.run_retrieval_eval baseline
```

Reports two sets of metrics and saves them to `eval/results_<tag>.json`:

- **Item-level** (`retrieval_golden.jsonl`): `recall@5` (did the correct
  item appear anywhere in the top 5), `hit@1` (was it ranked first), `mrr`
  (mean reciprocal rank -- rewards ranking it higher, not just finding it)
- **Category-level** (`retrieval_golden_vague.jsonl`): `category_precision@5`
  (of the 5 results returned for a vague query, what fraction were the
  right `product_type`)

Pass a different tag for each variant you want to compare (a prompt
change, a different rerank model, a different Groq model) -- each run
writes its own `results_<tag>.json` so runs can be diffed side by side:

```
python -m eval.run_retrieval_eval some_variant
```

### 3. Investigate failures by language

```
python -m eval.analyze_language_failures
```

Splits the item-level golden set into products whose embedded chunk text
(`name + brand + color + material + bullet_points`, see
`embeddings/chunker.py`) is fully English vs. products with non-English
`bullet_points`, and reports the miss rate for each group plus sample
failing queries -- useful for checking whether retrieval quality is weaker
on the ~24% of the catalog with non-English listing data.