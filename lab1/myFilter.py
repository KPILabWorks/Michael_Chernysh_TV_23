def my_filter(func, iterable):
    return (item for item in iterable if func(item))
