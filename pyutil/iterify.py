from typing import Any, Generator, Hashable, Iterable, List, Set, Union


def chunkify(obj: Iterable[Any], chunk_size: int) -> List[Any]:
    """Split the given iterable object into chunks

    Parameters
    ----------
    obj: Iterable[Any]
        The object to be chunkified
    chunk_size: int
        The size of each chunk
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

    Parameters
    ----------
    obj: Iterable[Any]
        The object to be chunkified
    """
    if isinstance(obj, str) or not hasattr(obj, "__iter__"):
        yield obj
    else:
        for objs in obj:
            yield from flatten(objs)


def listify(obj: Union[Iterable[Any], Any]) -> List[Any]:
    """Turn a given iterable object into a list

    Parameters
    ----------
    obj: Union[Iterable[Any], Any]
        The object to be lisified
    """
    if isinstance(obj, str) or not hasattr(obj, "__iter__"):
        return [obj]
    else:
        return list(obj)


def setify(obj: Union[Iterable[Hashable], Hashable]) -> Set[Hashable]:
    """Turn a given iterable object into a set, the element in the iterable should be hashable

    Parameters
    ----------
    obj: Union[Iterable[Hashable], Hashable]
        The object to be setified
    """
    if isinstance(obj, str) or not hasattr(obj, "__iter__"):
        return {obj}
    else:
        return set(obj)


def uniquify(obj: Union[Iterable[Hashable], Hashable]) -> List[Hashable]:
    """Uniquify a given iterable object, the element in the iterable should be hashable

    Parameters
    ----------
    obj: Union[Iterable[Hashable], Hashable]
        The object to be lisified
    """
    if isinstance(obj, str) or not hasattr(obj, "__iter__"):
        return list({obj})
    return list(setify(obj))
