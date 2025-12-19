from typing import List
import numpy as np


def preprocess_features(features: List[float]):
    """
    Adapt this to match the preprocessing you used during training.
    Right now it's just converting to a 2D numpy array.
    """
    X = np.array(features, dtype=float).reshape(1, -1)
    # TODO: apply scaling/encoding if used them in training
    return X
