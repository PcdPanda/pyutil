import time
import pandas as pd
from typing import Callable, Dict, List


class Wrapped(object):
    def __init__(self, func: Callable, info: Dict[str, List[float]]):
        name = func.__name__
        if name not in info.keys():
            info[name] = list()
        self.records = info[name]
        self.func = func
        
    def __call__(self, *args, **kwargs):
        t = time.time()
        res = self.func(*args, **kwargs)
        self.records.append(time.time() - t)
        return res
    
            
class Timer(object):
    '''A decorator class to profiling the time usage of functions
    
    Examples:
    # Decorate a function
    >>> @Timer.wrap
    >>> def my_func():
    >>>     pass
    
    # Decorate a method
    >>> cls.func = Timer.wrap(cls)

    '''
    info = dict()  

    @classmethod
    def wrap(cls, func: Callable) -> Wrapped:
        return func if isinstance(func, Wrapped) else Wrapped(func, cls.info)
    
    @classmethod
    def get_stats(cls) -> pd.DataFrame:
        stats = list()
        for func, rel in cls.info.items():
            rel = pd.Series(rel)
            stats.append(dict(func=func, count=len(rel), mean=rel.mean(), max=rel.max(), min=rel.min(), sum=rel.sum()))
        return pd.DataFrame(stats)
    
    @classmethod
    def reset(cls):
        cls.info.clear()
