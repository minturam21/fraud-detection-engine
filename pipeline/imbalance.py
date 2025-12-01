import pandas as pd
from sklearn.utils import resample

# class distribution
def get_class_distribution(y):
    """Return counts for each class in y as a dict.
    Example: {0: 990, 1: 10}"""

    counts = pd.Series(y).value_counts().to_dict()
    return {
        0: int(counts.get(0,0)),
        1: int(counts.get(1,0))
    }

# class weight
def compute_class_weight_dict(y):
    """Compute simple class weights:
    weight[class] = total_samples / (2 * class_count)
    This increases weight for fraud class when fraud is rare."""

    y = pd.Series(y)
    total = len(y)
    counts = y.value_counts().to_dict()

    weights={}
    for cls in [0,1]:
        count_cls = counts.get(cls,0)
        if count_cls ==0:
            weights[cls] =0.0
        else:
            weights[cls] = float(total)/(2.0*count_cls)
    return weights

# oversampling
def oversampling_minority(X,y, random_state=42):
    """Random oversampling of minority class.
    Duplicates minority rows until both classes have equal count.

    Returns:
        X_resampled, y_resampled"""
    
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    df = X.copy()
    df["label"] = y

    class_counts = df["label"].value_counts()
    if len(class_counts)<=1:
        return X,y
    
    majority_count = class_counts.max()

    result_frames = []
    for cls, count in class_counts.items():  
        cls_df = df[df["label"]==cls]

    # oversampling minority
    if count < majority_count:  
        cls_df_res = resample(
            cls_df,
            replace=True,
            n_samples=majority_count,
            random_state=random_state
        )
    else:
        cls_df_res=cls_df

    result_frames.append(cls_df_res)    

    df_resampled=(
        pd.concat(result_frames, ignore_index=True)  
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    y_res = df_resampled["label"].astype(int)    
    X_res = df_resampled.drop(columns=["label"])  

    return X_res, y_res


# main public api
def imbalance(X,y, method ="class_weight", random_state=42):
    """Main entry point for imbalance handling.
    Parameters:
        X      : DataFrame of features
        y      : labels (Series or list-like of 0/1)
        method : "class_weight", "oversample", or "none"
    Returns:
        X_resampled, y_resampled, class_weight_dict_or_None

    Behavior:
      - "class_weight": returns original X,y and class_weight dict
      - "oversample"  : returns oversampled X,y and None
      - "none"        : returns original X,y and None"""
    
    method = method.lower()
    if method not in {"class_weight", "oversample", "none"}:
        raise ValueError("method must be in class_weight, oversample, none")
    y_series = pd.Series(y)
    x_df = pd.DataFrame(X)

    # no imbalance handling
    if method =="none":
        return x_df, y_series, None
    
    # class_weight methode
    if method =="class_weight":
        cw = compute_class_weight_dict(y_series)
        return x_df, y_series, cw
    
    # oversample minority class
    if method =="oversample":
        X_res, y_res = oversampling_minority(x_df, y_series, random_state=random_state)

        return X_res, y_res,None
    
    return x_df, y_series, None