import os
import json
import numpy as np
import pandas as pd
from typing import Tuple
from typing import List

def load_index(data_dir: str = "data") -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    df = pd.read_parquet(os.path.join(data_dir, "library.parquet"))
    vecs = np.load(os.path.join(data_dir, "vectors.npy"))
    ids  = json.load(open(os.path.join(data_dir, "id_index.json"), "r"))
    return df, vecs, ids
