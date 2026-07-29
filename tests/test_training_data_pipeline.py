"""Tests for target-date splits and train-only scaling."""

from __future__ import annotations

from datetime import date
import unittest

import numpy as np
import pandas as pd

from src.training.data_pipeline import (
    DatasetSplitConfig,
    prepare_training_splits,
)


FEATURES = ("feature_a", "feature_b")
SPLIT = DatasetSplitConfig(
    train_end=date(2020, 6, 30),
    validation_end=date(2020, 12, 31),
    test_end=date(2021, 6, 30),
)


def sample_dataset(validation_shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for episode_id, dates, shift in (
        (
            "2020-H1",
            pd.bdate_range("2020-01-02", "2020-06-30"),
            0.0,
        ),
        (
            "2020-H2",
            pd.bdate_range("2020-07-01", "2020-12-31"),
            validation_shift,
        ),
        (
            "2021-H1",
            pd.bdate_range("2021-01-04", "2021-06-30"),
            0.0,
        ),
    ):
        for index, target_date in enumerate(dates):
            rows.append(
                {
                    "target_date": target_date,
                    "episode_id": episode_id,
                    "episode_complete": True,
                    "feature_a": float(index) + shift,
                    "feature_b": float(index % 7) - shift,
                }
            )
    return pd.DataFrame(rows)


class SplitTests(unittest.TestCase):
    def test_splits_by_target_date_without_crossing_episodes(self) -> None:
        splits, _ = prepare_training_splits(
            sample_dataset(),
            feature_columns=FEATURES,
            split_config=SPLIT,
        )

        self.assertEqual(
            set(splits["train"]["episode_id"]), {"2020-H1"}
        )
        self.assertEqual(
            set(splits["validation"]["episode_id"]), {"2020-H2"}
        )
        self.assertEqual(
            set(splits["test"]["episode_id"]), {"2021-H1"}
        )

    def test_validation_outlier_does_not_change_scaler(self) -> None:
        _, baseline = prepare_training_splits(
            sample_dataset(validation_shift=0.0),
            feature_columns=FEATURES,
            split_config=SPLIT,
        )
        _, shifted = prepare_training_splits(
            sample_dataset(validation_shift=1_000_000.0),
            feature_columns=FEATURES,
            split_config=SPLIT,
        )

        self.assertEqual(baseline.means, shifted.means)
        self.assertEqual(baseline.scales, shifted.scales)

    def test_training_features_are_standardized(self) -> None:
        splits, _ = prepare_training_splits(
            sample_dataset(),
            feature_columns=FEATURES,
            split_config=SPLIT,
        )
        values = splits["train"].loc[:, FEATURES].to_numpy()

        np.testing.assert_allclose(values.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(values.std(axis=0), 1.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
