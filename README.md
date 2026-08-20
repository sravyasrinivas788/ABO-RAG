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
