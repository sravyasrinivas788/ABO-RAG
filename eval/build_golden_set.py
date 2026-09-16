

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DATA_PATH = Path(__file__).parent.parent / "data" / "products_clean.jsonl"
OUTPUT_PATH = Path(__file__).parent / "retrieval_golden.jsonl"
VAGUE_OUTPUT_PATH = Path(__file__).parent / "retrieval_golden_vague.jsonl"

CAP_PER_CATEGORY = 8
MAX_CATEGORIES = 30  # keep only the N most frequent product_types
SEED = 42
GROQ_MODEL = "openai/gpt-oss-120b"

client = Groq()


def load_products_by_type(data_path: Path) -> dict[str, list[dict]]:
    by_type = defaultdict(list)
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p.get("name_language", "").startswith("en"):
                by_type[p["product_type"]].append(p)
    return by_type


def stratified_sample(by_type: dict[str, list[dict]], cap: int, seed: int, max_categories: int) -> list[dict]:
    random.seed(seed)
    top_types = sorted(by_type, key=lambda t: len(by_type[t]), reverse=True)[:max_categories]
    sample = []
    for ptype in top_types:
        items = by_type[ptype]
        sample.extend(random.sample(items, min(cap, len(items))))
    return sample


def template_query(p: dict) -> str:
    parts = [p.get("brand") or "", p.get("color") or "", (p.get("product_type") or "").replace("_", " ").lower()]
    return " ".join(x for x in parts if x).strip()


def natural_query(p: dict) -> str:
    prompt = f"""A shopper is searching a product catalog. Given this product, write ONE short, casual
search query (4-8 words) a real person might type to find it.

Pick only ONE distinguishing detail (e.g. just the color, OR just a style word, OR just the use
case) -- not the full combination of color + pattern + material + brand together. Do not copy the
product name verbatim. Do not include the exact brand name unless a shopper would realistically
search by brand. It should read like something typed into a search box in a hurry, not catalog copy.

Product name: {p.get('name')}
Brand: {p.get('brand')}
Type: {p.get('product_type')}
Bullets: {p.get('bullet_points')}

Return ONLY the query text, nothing else."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip().strip('"')


def vague_query(p: dict) -> str:
    """A deliberately under-specified, category-level query. Has NO single correct
    item_id -- eval this tier by product_type/category match in top-k, not exact recall."""
    prompt = f"""A shopper is browsing a product catalog, not looking for anything specific yet.
Given this product's category, write ONE very short, generic browsing query (2-4 words) that
would plausibly return this AND many similar products -- think "pillow options" or "cheap phone
cases", not anything that names a specific color, pattern, or brand.

Type: {p.get('product_type')}
Category: {p.get('category_primary')}

Return ONLY the query text, nothing else."""
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip().strip('"')


def build_golden_set(cap: int = CAP_PER_CATEGORY, seed: int = SEED, max_categories: int = MAX_CATEGORIES) -> tuple[list[dict], list[dict]]:
    by_type = load_products_by_type(DATA_PATH)
    sample = stratified_sample(by_type, cap, seed, max_categories)
    print(f"Sampled {len(sample)} products across {min(max_categories, len(by_type))} of {len(by_type)} categories")

    golden = []
    seen_types = set()
    vague_golden = []
    for i, p in enumerate(sample, start=1):
        try:
            q_natural = natural_query(p)
        except Exception as e:
            print(f"  [{i}/{len(sample)}] skipping {p['item_id']} -- LLM call failed: {e}")
            continue

        golden.append({
            "item_id": p["item_id"],
            "product_type": p["product_type"],
            "query_template": template_query(p),
            "query_natural": q_natural,
        })
        print(f"  [{i}/{len(sample)}] {p['item_id']} ({p['product_type']}): {q_natural!r}")
        time.sleep(0.2)  # stay well under Groq rate limits

        # one vague/category-level query per product_type, not per item --
        # these have no single correct item_id, so generating per-item would
        # just waste calls on near-duplicate vague phrasing within a category.
        if p["product_type"] not in seen_types:
            seen_types.add(p["product_type"])
            try:
                q_vague = vague_query(p)
                vague_golden.append({
                    "product_type": p["product_type"],
                    "query_vague": q_vague,
                })
                print(f"    vague query for {p['product_type']}: {q_vague!r}")
                time.sleep(0.2)
            except Exception as e:
                print(f"    skipping vague query for {p['product_type']} -- LLM call failed: {e}")

    return golden, vague_golden


def main():
    golden, vague_golden = build_golden_set()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in golden:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(golden)} item-level rows to {OUTPUT_PATH}")

    with open(VAGUE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in vague_golden:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(vague_golden)} category-level (vague) rows to {VAGUE_OUTPUT_PATH}")

    print("\nNext: manually review both files -- for retrieval_golden.jsonl, fix/discard any query")
    print("that isn't uniquely answered by its item_id. For retrieval_golden_vague.jsonl, eval")
    print("success as 'are the top-k results the right product_type', not exact item_id match.")


if __name__ == "__main__":
    main()
