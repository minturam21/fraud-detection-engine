import pandas as pd
import logging
from .clean import data_cleaned as clean_data
from .features import assemble_features, select_model_features
from .temporal_split import temporal_train_text_split
from .train_model import train_model
from .evaluate import evaluate_model
from .save_model import save_trained_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Automatically pick a time-based split date
def auto_split_date(df, time_col="timestamp"):
    ts = pd.to_datetime(df[time_col], errors="coerce").sort_values().reset_index(drop=True)
    cutoff = int(len(ts) * 0.8)   # 80% for training
    return ts.iloc[cutoff]

# Main pipeline
def main():

    # Load
    data_path = "data/synthetic/transactions.csv"
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows.")

    # Clean
    logger.info("Cleaning data...")
    df_clean = clean_data(df)

    # Feature engineering
    logger.info("Assembling features...")
    df_features = assemble_features(df_clean)

    # Choose split date BEFORE dropping timestamp
    split_date = auto_split_date(df_features, time_col="timestamp")
    logger.info(f"Auto-selected split date: {split_date}")

    # Temporal split (needs timestamp)
    X_train_raw, X_test_raw, y_train, y_test = temporal_train_text_split(
        df_features,
        split_date=split_date,
        label_col="label"
    )

    # Drop identifiers AFTER splitting
    X_train = select_model_features(X_train_raw)
    X_test = select_model_features(X_test_raw)

    # Train
    logger.info("Training model...")
    model, test_probs, class_weight = train_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model_type="lightgbm",
        imbalance_method="class_weight"
    )

    # Evaluate
    logger.info("Evaluating model...")
    metrics = evaluate_model(y_test, test_probs)
    logger.info(f"Evaluation metrics: {metrics}")

    # Save model + metadata + thresholds
    save_trained_model(model, class_weight, metrics)

    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
