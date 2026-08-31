from retrival.semantic_search import semantic_search

results = semantic_search("something to stop a drawer from sticking", top_k=20)
for i, r in enumerate(results, 1):
    marker = " <-- DRAWER SLIDES" if r["item_id"] == "B07P8ML82R" else ""
    print(f"{i}. {r['item_id']}  score={r['score']:.4f}{marker}")