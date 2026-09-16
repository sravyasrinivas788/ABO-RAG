"""Checks whether item-level retrieval misses correlate with non-English
bullet_points text (the embedded chunk text is name+brand+color+material+
bullets, so a product can have an English name but foreign-language bullets
that dominate its embedding -- see embeddings/chunker.py).

Run:
    python -m eval.analyze_language_failures
"""

import json
from pathlib import Path
from collections import defaultdict

from retrival.hybrid_search import hybrid_search as search_fn

GOLDEN_PATH = Path(__file__).parent / "retrieval_golden.jsonl"
DATA_PATH = Path(__file__).parent.parent / "data" / "products_clean.jsonl"


def load_products_by_id() -> dict[str, dict]:
    by_id = {}
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            by_id.setdefault(p["item_id"], p)  # keep first occurrence for the ~9 duplicate item_ids
    return by_id


def main(k: int = 5):
    golden = [json.loads(l) for l in open(GOLDEN_PATH, encoding="utf-8")]
    products_by_id = load_products_by_id()

    groups = defaultdict(lambda: {"hit": 0, "miss": 0, "misses_detail": []})

    for row in golden:
        product = products_by_id.get(row["item_id"])
        if product is None:
            continue

        bullets_lang = product.get("bullets_language") or "none"
        group = "english_bullets" if bullets_lang.startswith("en") or bullets_lang == "none" else "non_english_bullets"

        results = search_fn(row["query_natural"], top_k=k)
        retrieved_ids = [r["item_id"] for r in results]
        hit = row["item_id"] in retrieved_ids

        groups[group]["hit" if hit else "miss"] += 1
        if not hit:
            groups[group]["misses_detail"].append({
                "item_id": row["item_id"],
                "product_type": row["product_type"],
                "bullets_language": bullets_lang,
                "query_natural": row["query_natural"],
            })

    print(f"{'group':<22} {'n':>5} {'misses':>7} {'miss_rate':>10}")
    for group, stats in groups.items():
        n = stats["hit"] + stats["miss"]
        miss_rate = stats["miss"] / n if n else 0
        print(f"{group:<22} {n:>5} {stats['miss']:>7} {miss_rate:>10.1%}")

    print("\nSample misses from non_english_bullets group:")
    for m in groups["non_english_bullets"]["misses_detail"][:10]:
        print(f"  {m['item_id']} ({m['product_type']}, bullets={m['bullets_language']}): {m['query_natural']!r}")

    print("\nSample misses from english_bullets group:")
    for m in groups["english_bullets"]["misses_detail"][:10]:
        print(f"  {m['item_id']} ({m['product_type']}): {m['query_natural']!r}")


if __name__ == "__main__":
    main()
