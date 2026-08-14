from dataclasses import dataclass


@dataclass
class ValueResult:
    value: object          
    is_missing: bool


@dataclass
class CategoryResult:
    primary_name: str | None
    primary_id: int | None
    all_names: list[str]
    is_missing: bool


def get_value_only(field: list | None) -> ValueResult:
    if not field:
        return ValueResult(value=None, is_missing=True)
    return ValueResult(value=field[0].get("value"), is_missing=False)


def get_categories(node_field: list | None) -> CategoryResult:
    """For node -- CAN have multiple entries (real data shows up to 10)."""
    if not node_field:
        return CategoryResult(primary_name=None, primary_id=None, all_names=[], is_missing=True)

    names = [n.get("node_name") for n in node_field if n.get("node_name")]
    return CategoryResult(
        primary_name=node_field[0].get("node_name"),
        primary_id=node_field[0].get("node_id"),
        all_names=names,
        is_missing=False,
    )


