from pydantic import BaseModel

from ingestion.normalize_text import get_localized_single, get_localized_list
from ingestion.normalize_fields import get_value_only, get_categories
from ingestion.normalize_measurement import extract_dimensions, extract_weight
from ingestion.image_urls import extract_images


class Product(BaseModel):
    item_id: str
    name: str | None
    name_language: str | None
    name_is_fallback: bool

    brand: str | None
    color: str | None
    material: str | None
    bullet_points: list[str]
    bullets_language: str | None

    product_type: str | None
    model_number: str | None

    category_primary: str | None
    category_all: list[str]

    height_in: float | None
    length_in: float | None
    width_in: float | None
    weight_lb: float | None

    main_image_url: str | None
    other_image_urls: list[str]
    has_no_image: bool

    embedding_text: str  

def build_product(item:dict)->BaseModel:
    name = get_localized_single(item.get("item_name"))
    brand = get_localized_single(item.get("brand"))
    color = get_localized_single(item.get("color"))
    material = get_localized_single(item.get("material"))
    bullets = get_localized_list(item.get("bullet_point"))

    product_type = get_value_only(item.get("product_type"))
    model_number = get_value_only(item.get("model_number"))
    categories = get_categories(item.get("node"))

    dims = extract_dimensions(item.get("item_dimensions"))
    weight = extract_weight(item.get("item_weight"))

    images = extract_images(item)

    text_parts = [p for p in [name.value, brand.value, color.value, material.value] if p]
    text_parts.extend(b for b in bullets.value if b)
    embedding_text = " | ".join(text_parts)

    return Product(
        item_id=item["item_id"],
        name=name.value,
        name_language=name.language_used,
        name_is_fallback=name.is_fallback,
        brand=brand.value,
        color=color.value,
        material=material.value,
        bullet_points=bullets.value,
        bullets_language=bullets.language_used,
        product_type=product_type.value,
        model_number=model_number.value,
        category_primary=categories.primary_name,
        category_all=categories.all_names,
        height_in=dims["height"].value,
        length_in=dims["length"].value,
        width_in=dims["width"].value,
        weight_lb=weight.value,
        main_image_url=images.main_url,
        other_image_urls=images.other_urls,
        has_no_image=images.has_no_image,
        embedding_text=embedding_text,
    )

if __name__ == "__main__":
    # Test with the real drawer-slide listing (Spanish-only, has dimensions+weight, clean image)
    drawer_item = {
        "item_id": "B07P8ML82R",
        "item_name": [{"language_tag": "es_MX", "value": '22" Bottom Mount Drawer Slides, White Powder Coat, 10 Pairs'}],
        "brand": [{"language_tag": "es_MX", "value": "AmazonBasics"}],
        "bullet_point": [
            {"language_tag": "es_MX", "value": "White Powder Coat Finish"},
            {"language_tag": "es_MX", "value": "55-Lbs max weight capacity"},
        ],
        "product_type": [{"value": "HARDWARE"}],
        "model_number": [{"value": "AB5013-R22-10"}],
        "node": [{"node_id": 9827962011, "node_name": "/Categorías/Ferretería/Guías para Cajones"}],
        "item_dimensions": {
            "height": {"normalized_value": {"unit": "inches", "value": 0.9}},
            "length": {"normalized_value": {"unit": "inches", "value": 22}},
            "width": {"normalized_value": {"unit": "inches", "value": 0.87}},
        },
        "item_weight": [{"normalized_value": {"unit": "pounds", "value": 1.45}}],
        "main_image_id": "619y9YG9cnL",
        "other_image_id": ["51Fqps5k9YL"],
    }

    p = build_product(drawer_item)
    print(p.model_dump_json(indent=2))

    assert p.name_is_fallback is True and p.name_language == "es_MX"
    assert p.length_in == 22 and p.weight_lb == 1.45
    assert p.main_image_url == "https://m.media-amazon.com/images/I/619y9YG9cnL.jpg"
    assert p.has_no_image is False
    assert "Drawer Slides" in p.embedding_text and "AmazonBasics" in p.embedding_text

    print("\nAll Phase 5 tests passed.")