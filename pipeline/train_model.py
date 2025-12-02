import pandas as pd
import numpy as np
from .imbalance import imbalance

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False


# Build the LightGBM model with class weights
def build_model(model_type="lightgbm", class_weight=None):
    model_type = model_type.lower()

    if model_type == "lightgbm" and LGB_AVAILABLE:
        # Convert class_weight dict - LightGBM scale_pos_weight
        scale_pos_weight = None
        if class_weight is not None and 1 in class_weight and 0 in class_weight:
            scale_pos_weight = class_weight[1] / (class_weight[0] + 1e-6)

        return lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight
        )

    raise ValueError("LightGBM not available or unsupported model type.")

# Train the model (no split happens here)

def train_model(
    X_train,
    X_test,
    y_train,
    y_test,
    model_type="lightgbm",
    imbalance_method="class_weight"
):

    # Handle imbalance
    X_train_bal, y_train_bal, class_weight = imbalance(
        X_train,
        y_train,
        method=imbalance_method
    )

    # Build model
    model = build_model(model_type=model_type, class_weight=class_weight)

    # Train model
    model.fit(X_train_bal, y_train_bal)

    # Predict probabilities on test set
    if hasattr(model, "predict_proba"):
        test_probs = model.predict_proba(X_test)[:, 1]
    else:
        test_probs = model.predict(X_test)

    return model, test_probs, class_weight
