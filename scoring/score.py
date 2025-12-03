import json
import joblib
import logging
import pandas as pd
from pathlib import Path

from pipeline.clean import data_cleaned
from pipeline.features import assemble_features
from pipeline.features import select_model_features
from .rule_engine import decision_pipeline
from scoring.decision_pipeline import decision_pipeline



# Logger 
logger = logging.getLogger("scoring")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | scoring | %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Scoring Engine 
class ScoringEngine:

    def __init__(self, model_path="data/models/model.pkl", threshold_path="data/models/thresholds.json"):
        logger.info("Initializing scoring engine")

        self.model = joblib.load(model_path)
        logger.info(f"Loaded model: {model_path}")

        if Path(threshold_path).exists():
            with open(threshold_path, "r") as f:
                self.threshold = json.load(f)
            logger.info(f"Loaded thresholds: {threshold_path}")
        else:
            logger.warning("Threshold file missing, using default.")
            self.threshold = {"low": 0.2, "medium": 0.5, "high": 0.8}

        # IMPORTANT - read training feature names from model
        self.train_features = self.model.feature_name_


    def score(self, df: pd.DataFrame):
        logger.info(f"Scoring {len(df)} rows...")

        df_clean = data_cleaned(df)
        df_features = assemble_features(df_clean)

        # Extract model features only
        df_model = select_model_features(df_features)

        # FIX: force model feature alignment
        df_model = df_model.reindex(columns=self.train_features, fill_value=0)

        # Model prediction
        model_score = self.model.predict_proba(df_model)[:, 1]

        # Rule engine score (simple: sum of rule-trigger values if you add rule scoring later)
        rule_score = 0.0
        rule_flags = []

        final_scores = model_score  # can later merge model+rule

        results = []
        for ms, rs, flags in zip(model_score, [rule_score]*len(df_model), [rule_flags]*len(df_model)):
            decision = decision_pipeline(
                final_score=ms,
                model_score=ms,
                rule_score=rs,
                rule_flags=flags,
                threshold=self.threshold
            )
            results.append(decision)

        return results


# Run Standalone 
if __name__ == "__main__":
    engine = ScoringEngine()

    sample = pd.read_csv("data/synthetic/transactions.csv").tail(5)
    results = engine.score(sample)

    for r in results:
        print(json.dumps(r, indent=4))
