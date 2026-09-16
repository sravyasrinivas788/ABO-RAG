
import json
import sys
from pathlib import Path

from retrival.hybrid_search import hybrid_search as search_fn
from vector_store.qdrant_store import get_client, COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue

GOLDEN_PATH = Path(__file__).parent / "retrieval_golden.jsonl"
VAGUE_GOLDEN_PATH = Path(__file__).parent / "retrieval_golden_vague.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate_item_level(golden: list[dict], k: int = 5) -> dict:
    hits_at_k = 0
    hits_at_1 = 0
    reciprocal_ranks = []

    for row in golden:
        results = search_fn(row["query_natural"], top_k=k)
        retrieved_ids = [r["item_id"] for r in results]

        if row["item_id"] in retrieved_ids:
            hits_at_k += 1
            rank = retrieved_ids.index(row["item_id"]) + 1
            reciprocal_ranks.append(1 / rank)
            if rank == 1:
                hits_at_1 += 1
        else:
            reciprocal_ranks.append(0)

    n = len(golden)
    return {
        "n": n,
        f"recall@{k}": hits_at_k / n,
        "hit@1": hits_at_1 / n,
        "mrr": sum(reciprocal_ranks) / n,
    }


def get_product_type(client, item_id: str) -> str | None:
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="item_id", match=MatchValue(value=item_id))]),
        limit=1,
    )
    return points[0].payload.get("product_type") if points else None


def evaluate_category_level(vague_golden: list[dict], k: int = 5) -> dict:
    client = get_client()
    precisions = []

    for row in vague_golden:
        results = search_fn(row["query_vague"], top_k=k)
        if not results:
            precisions.append(0)
            continue
        matched = sum(1 for r in results if get_product_type(client, r["item_id"]) == row["product_type"])
        precisions.append(matched / len(results))

    return {
        "n": len(vague_golden),
        f"category_precision@{k}": sum(precisions) / len(precisions) if precisions else 0,
    }


def main(config_tag: str = "baseline", k: int = 5):
    golden = load_jsonl(GOLDEN_PATH)
    vague_golden = load_jsonl(VAGUE_GOLDEN_PATH)

    print(f"Running eval with config={config_tag!r}")
    item_metrics = evaluate_item_level(golden, k=k)
    category_metrics = evaluate_category_level(vague_golden, k=k)

    print(f"\n--- config: {config_tag} ---")
    print("Item-level:", item_metrics)
    print("Category-level:", category_metrics)

    out_path = Path(__file__).parent / f"results_{config_tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": config_tag, "item_level": item_metrics, "category_level": category_metrics}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    main(config_tag=tag)
