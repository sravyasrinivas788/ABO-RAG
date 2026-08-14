

from dataclasses import dataclass


@dataclass
class MeasurementResult:
    value: float | None
    unit: str | None
    is_missing: bool


def extract_measurement(field: dict | None) -> MeasurementResult:
    if not field:
        return MeasurementResult(value=None, unit=None, is_missing=True)

    normalized = field.get("normalized_value")
    if normalized and normalized.get("value") is not None:
        return MeasurementResult(value=normalized["value"], unit=normalized.get("unit"), is_missing=False)

    # normalized_value missing but raw value present -- fall back, but this
    # means the unit is NOT guaranteed consistent with other products
    if field.get("value") is not None:
        return MeasurementResult(value=field["value"], unit=field.get("unit"), is_missing=False)

    return MeasurementResult(value=None, unit=None, is_missing=True)


def extract_dimensions(item_dimensions: dict | None) -> dict[str, MeasurementResult]:
    """item_dimensions is a plain dict (NOT a list) with height/length/width keys."""
    if not item_dimensions:
        missing = MeasurementResult(value=None, unit=None, is_missing=True)
        return {"height": missing, "length": missing, "width": missing}

    return {
        "height": extract_measurement(item_dimensions.get("height")),
        "length": extract_measurement(item_dimensions.get("length")),
        "width": extract_measurement(item_dimensions.get("width")),
    }


def extract_weight(item_weight: list | None) -> MeasurementResult:
    """item_weight IS a list, unlike item_dimensions -- confirmed always length 1 in real data."""
    if not item_weight:
        return MeasurementResult(value=None, unit=None, is_missing=True)
    return extract_measurement(item_weight[0])


