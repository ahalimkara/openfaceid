from typing import Any, List


def is_str_list(items: List[Any]) -> bool:
    return all(isinstance(item, str) for item in items)


def is_float_list(items: List[Any]) -> bool:
    return all(isinstance(item, float) for item in items)
