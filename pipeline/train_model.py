import pandas as pd
import numpy as np
from .temporal_split import temporal_train_text_split
from .imbalance import imbalance


try:
    import lightgbm as lgb
    LGB_AVAIABLE = True
    print("working ✅")
except Exception:
    LGB_AVAIABLE = False


def build_model(model_type="lightbgm", class_weight =None):
    model_type = model_type.lower()
    if model_type == "lightbgm" and LGB_AVAIABLE:
        # Convert class_weight dict- scale_pos_weight
        scale_pos_weight=None
        if class_weight is not None and 1 in class_weight:
            scale_pos_weight = class_weight[1]/(class_weight[0] + 1e-6)
    
    return lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight
    )

def train_model( df: pd.DataFrame, split_date: str, imbalance_method= "class_weight", model_type="lightgbm"):

    # temporal split
    X_train, X_test, y_train, y_test = temporal_train_text_split(
        df,
        split_date=split_date,
        label_col="label"
    )

    # handle imbalance
    X_train_final, y_train_final, class_weight = imbalance(
        X_train, y_train,
        method=imbalance_method
    )

    model = build_model(model_type=model_type, class_weight=class_weight)
    model.fit(X_train_final,y_train_final)

    if hasattr(model, "predict_proba"):
        test_probs = model.predict_proba(X_test)[:,1]
    else:
        test_probs = model.predict(X_test)

    return model, test_probs, class_weight,(X_train, X_test, y_train, y_test)


    
    

    