from typing import List
from utils.model_loader import get_model
from utils.preprocess import preprocess_features


class ModelPipeline:
    """
    Wraps pure ML model scoring:
      - loads the model once
      - preprocesses features
      - returns a risk score in [0, 1] (you can adapt)
    """

    def __init__(self):
        self.model = get_model()

    def score(self, features: List[float]) -> float:
        """
        Return a numeric score using the underlying ML model.
        Adapt according to classifier/regressor behavior.
        """
        X = preprocess_features(features)

        # If classifier with predict_proba
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            # assumes positive class is at index 1
            return float(proba[0][1])

        # Else regressors or plain predict
        pred = self.model.predict(X)
        if hasattr(pred, "__len__"):
            return float(pred[0])
        return float(pred)
