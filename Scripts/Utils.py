def find_max(lst: list):
    _max = lst[0]
    for _item in lst:
        _max = _item if _item > _max else _max
    return _max
