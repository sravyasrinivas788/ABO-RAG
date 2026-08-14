import json
from  pathlib import Path 
from collections.abc import Iterator

DEFAULT_PATH=Path(__file__).parent.parent / "data" / "listings_0.json"

def load_listings(path: str| Path=DEFAULT_PATH)->Iterator[dict]:
    path=Path(path)
    with open(path,encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH

    count = 0
    first_item = None
    for item in load_listings(path):
        if first_item is None:
            first_item = item
        count += 1

    print(f"Loaded {count} listings from {path}")
    print(f"First item_id: {first_item.get('item_id')}")
    print(f"First item keys: {list(first_item.keys())}")