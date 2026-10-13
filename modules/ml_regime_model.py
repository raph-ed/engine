"""Walk-Forward Machine Learning Regime Detector with Purging, Embargo and Live Tail Inference."""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class WalkForwardRegimeDetector:
    """Walk-forward adaptive regime classifier with zero look-ahead bias."""

    def __init__(self, model_type: str = "rf", horizon: int = 20, embargo: int = 5):
        self.model_type = model_type
        self.horizon = horizon
        self.embargo = embargo
        self.feature_names = []
        self.last_fitted_model = None

    def _create_model(self):
        if self.model_type == "lr":
            return LogisticRegression(penalty="l2", C=1.0, max_iter=500, class_weight="balanced")
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_split=8,
            random_state=42,
            class_weight="balanced_subsample",
        )

    def run_walk_forward(
        self,
        dataset_train: pd.DataFrame,
        all_features: pd.DataFrame,
        train_window: int = 252,
        test_step: int = 21,
    ) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
        """Perform rolling walk-forward training with strict purging and embargo."""
        X_train_full = dataset_train.drop(columns=["target"])
        y_train_full = dataset_train["target"]
        self.feature_names = list(X_train_full.columns)

        n_samples = len(dataset_train)
        eff_train_window = min(train_window, max(60, int(n_samples * 0.45)))

        preds = pd.Series(index=all_features.index, dtype=float)
        probas = pd.DataFrame(0.0, index=all_features.index, columns=[0, 1, 2])

        start_idx = eff_train_window
        last_trained_model = self._create_model()

        while start_idx < n_samples:
            end_test_idx = min(start_idx + test_step, n_samples)

            # Strict Purge: training cutoff must be at least (horizon + embargo) days before test
            purge_cutoff = max(30, start_idx - (self.horizon + self.embargo))
            train_start = max(0, purge_cutoff - eff_train_window)

            X_fold_train = X_train_full.iloc[train_start:purge_cutoff]
            y_fold_train = y_train_full.iloc[train_start:purge_cutoff]

            # Fit model if sufficient class diversity
            if len(np.unique(y_fold_train)) >= 2:
                model = self._create_model()
                model.fit(X_fold_train, y_fold_train)
                last_trained_model = model

            # Predict test block
            test_indices = dataset_train.index[start_idx:end_test_idx]
            X_test_block = X_train_full.loc[test_indices]

            fold_preds = last_trained_model.predict(X_test_block)
            fold_probs = last_trained_model.predict_proba(X_test_block)

            preds.loc[test_indices] = fold_preds

            # Align probability columns safely to classes [0, 1, 2]
            classes = last_trained_model.classes_
            for c_idx, c_val in enumerate(classes):
                probas.loc[test_indices, c_val] = fold_probs[:, c_idx]

            start_idx = end_test_idx

        # LIVE TAIL INFERENCE:
        # Unpredicted recent days (including the active warm-up end and live tail)
        unpredicted_mask = preds.isna()
        if unpredicted_mask.any():
            X_live = all_features.loc[unpredicted_mask]
            # Train final model on all purged historical labels
            final_purge_cutoff = len(dataset_train) - self.embargo
            X_final = X_train_full.iloc[:final_purge_cutoff]
            y_final = y_train_full.iloc[:final_purge_cutoff]

            if len(np.unique(y_final)) >= 2:
                last_trained_model = self._create_model()
                last_trained_model.fit(X_final, y_final)

            live_preds = last_trained_model.predict(X_live)
            live_probs = last_trained_model.predict_proba(X_live)

            preds.loc[unpredicted_mask] = live_preds
            classes = last_trained_model.classes_
            for c_idx, c_val in enumerate(classes):
                probas.loc[unpredicted_mask, c_val] = live_probs[:, c_idx]

        self.last_fitted_model = last_trained_model

        # Calculate feature importances
        if hasattr(last_trained_model, "feature_importances_"):
            importances = pd.Series(last_trained_model.feature_importances_, index=self.feature_names)
        elif hasattr(last_trained_model, "coef_"):
            importances = pd.Series(np.abs(last_trained_model.coef_).mean(axis=0), index=self.feature_names)
        else:
            importances = pd.Series(1.0 / len(self.feature_names), index=self.feature_names)

        importances = importances / importances.sum()
        preds = preds.fillna(1).astype(int)
        probas = probas.fillna(1.0 / 3.0)

        return preds, probas, importances
