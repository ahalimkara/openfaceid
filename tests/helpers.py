from typing import Any, List, TypeGuard


def is_str_list(items: List[Any]) -> TypeGuard[List[str]]:
    return all(isinstance(item, str) for item in items)


def is_float_list(items: List[Any]) -> TypeGuard[List[str]]:
    return all(isinstance(item, float) for item in items)
