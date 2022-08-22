from typing import Any, Generator, Hashable, Iterable, List, Set, Union


def chunkify(obj: Iterable[Any], chunk_size: int) -> List[Any]:
    """Split the given iterable object into chunks

    Args:
        obj (Iterable[Any]): The object to be chunkified

    Returns:
        (List[Any]): The chunkified object
    """
    chunks = list()
    if hasattr(obj, "__iter__"):
        for i in range(0, len(obj), chunk_size):
            chunks.append(obj[i:i + chunk_size])
        return chunks
    else:
        raise ValueError(f"The given obj type {type(obj)} is not iterable")


def flatten(obj: Union[Iterable[Any], Any]) -> Generator[Any, None, None]:
    """Flatten a given nested iterable object into a generator

    Args:
        obj (Iterable[Any]): The object to be chunkified

    Returns:
        (List[Any]): The chunkified object
    """
    if isinstance(obj, str) or not hasattr(obj, "__iter__"):
        yield obj
    else:
        for objs in obj:
            yield from flatten(objs)


def listify(obj: Union[Iterable[Any], Any]) -> List[Any]:
    if isinstance(obj, str) or not hasattr(obj, "__iter__"):
        return [obj]
    else:
        return list(obj)


def setify(obj: Union[Iterable[Hashable], Hashable]) -> Set[Hashable]:
    if isinstance(obj, str) or not hasattr(obj, "__iter__"):
        return {obj}
    else:
        return set(obj)


def uniquify(obj: Union[Iterable[Hashable], Hashable]) -> List[Hashable]:
    if isinstance(obj, str) or not hasattr(obj, "__iter__"):
        return list({obj})
    return list(setify(obj))
