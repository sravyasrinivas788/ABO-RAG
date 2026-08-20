

import time
from pathlib import Path

from ingestion.loader import load_listings
from ingestion.product_schema import build_product

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "products_clean.jsonl"


def materialize(input_path=None, output_path: Path = OUTPUT_PATH):
    start = time.time()
    total = 0
    errors = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        source = load_listings(input_path) if input_path else load_listings()
        for item in source:
            total += 1
            try:
                product = build_product(item)
                out.write(product.model_dump_json() + "\n")
            except Exception as e:
                errors += 1
                print(f"FAILED on {item.get('item_id')}: {e}")

    elapsed = time.time() - start
    print(f"Wrote {total - errors} clean products to {output_path}")
    print(f"Errors: {errors} / {total}")
    print(f"Time: {elapsed:.2f}s")
    return total - errors, errors


if __name__ == "__main__":
    materialize()