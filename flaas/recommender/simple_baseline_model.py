"""
Simple Baseline Model for FLaaS Recommender.
Implements a simple K-Nearest Neighbors / Ridge regressor baseline as requested
by the specification ("At most one simple model as an extra baseline. Do NOT build meta-learning or surrogate models").
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from flaas.recommender.scenario_matching import scenario_to_feature_vector
from flaas.grid.scenarios import SCENARIOS


class SimpleKNNBaseline:
    """
    Simple, non-neural baseline regressor that predicts configuration accuracy
    directly from scenario features.
    """

    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors
        self.models: Dict[str, KNeighborsRegressor] = {}
        self.scenarios_map = {s.scenario_id: s for s in SCENARIOS}

    def fit(self, domain_df: pd.DataFrame) -> None:
        """Fits an independent KNN regressor for each config_id in the domain dataset."""
        self.models.clear()
        config_ids = domain_df["config_id"].unique()

        for cid in config_ids:
            cdf = domain_df[domain_df["config_id"] == cid]
            X = []
            y = []
            for _, row in cdf.iterrows():
                sid = row["scenario_id"]
                if sid in self.scenarios_map:
                    s = self.scenarios_map[sid]
                    feat = scenario_to_feature_vector(
                        s.heterogeneity_alpha,
                        s.participation_rate,
                        s.attack_type,
                        s.attack_fraction,
                        s.attack_scale,
                        s.privacy_mode,
                    )
                    X.append(feat)
                    y.append(row["final_accuracy"])
            if len(X) >= 2:
                knn = KNeighborsRegressor(n_neighbors=min(self.n_neighbors, len(X)), weights="distance")
                knn.fit(np.array(X), np.array(y))
                self.models[cid] = knn

    def predict_best_config(self, query_feature: np.ndarray, candidate_config_ids: List[str]) -> str:
        """Predicts accuracy for each candidate and returns the config_id with highest predicted accuracy."""
        best_cid = candidate_config_ids[0]
        best_pred = -float("inf")

        for cid in candidate_config_ids:
            if cid in self.models:
                pred = float(self.models[cid].predict(query_feature.reshape(1, -1))[0])
            else:
                pred = 0.0
            if pred > best_pred:
                best_pred = pred
                best_cid = cid

        return best_cid
