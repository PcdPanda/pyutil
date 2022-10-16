from typing import Dict, Iterable, Union

import numpy as np
import pandas as pd

TensorLike = Union[Dict, float, int, np.ndarray, np.number, pd.DataFrame,
                   pd.Series, Iterable["TensorLike"]]
