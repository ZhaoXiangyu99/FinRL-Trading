#!/usr/bin/env python3
"""Run a small CPU PPO smoke test without touching the sealed test split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import gymnasium
import numpy as np
import pandas as pd
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environments.gym_qqq_spy_cash import QQQSpyCashGymEnv
from src.environments.qqq_spy_cash_env import RewardConfig
from src.evaluation.trained_policy import evaluate_model_on_episodes
from src.training.data_pipeline import prepare_training_splits


DEFAULT_CONFIG = Path(
    "configs/qqq_spy_agent/ppo_cpu_smoke_v1.json"
)
DEFAULT_DATASET = Path(
    "data/market/longbridge/qqq_spy_rl_training_dataset_v1.csv"
)
DEFAULT_DATASET_METADATA = DEFAULT_DATASET.with_suffix(
    DEFAULT_DATASET.suffix + ".metadata.json"
)
DEFAULT_OUTPUT = Path("artifacts/qqq_spy_agent/cpu_smoke_v1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episodes(frame: pd.DataFrame) -> list[pd.DataFrame]:
    return [
        episode.reset_index(drop=True)
        for _, episode in frame.groupby("episode_id", sort=True)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPU-only PPO smoke training on the training split."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-metadata",
        type=Path,
        default=DEFAULT_DATASET_METADATA,
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset_metadata = json.loads(
        args.dataset_metadata.read_text(encoding="utf-8")
    )
    feature_columns = dataset_metadata["feature_columns"]
    dataset = pd.read_csv(
        args.dataset,
        parse_dates=["feature_date", "target_date"],
    )
    splits, _ = prepare_training_splits(
        dataset, feature_columns=feature_columns
    )

    reward = RewardConfig(**config["reward"])
    transaction_cost_rate = float(config["transaction_cost_rate"])
    train_env = QQQSpyCashGymEnv(
        episodes=episodes(splits["train"]),
        feature_columns=feature_columns,
        transaction_cost_rate=transaction_cost_rate,
        reward_config=reward,
        episode_selection="random",
    )
    validation_env = QQQSpyCashGymEnv(
        episodes=episodes(splits["validation"]),
        feature_columns=feature_columns,
        transaction_cost_rate=transaction_cost_rate,
        reward_config=reward,
        episode_selection="sequential",
    )
    check_env(train_env, warn=True, skip_render_check=True)

    ppo_config = config["ppo"]
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=float(ppo_config["learning_rate"]),
        n_steps=int(ppo_config["n_steps"]),
        batch_size=int(ppo_config["batch_size"]),
        n_epochs=int(ppo_config["n_epochs"]),
        gamma=float(ppo_config["gamma"]),
        gae_lambda=float(ppo_config["gae_lambda"]),
        clip_range=float(ppo_config["clip_range"]),
        ent_coef=float(ppo_config["ent_coef"]),
        vf_coef=float(ppo_config["vf_coef"]),
        max_grad_norm=float(ppo_config["max_grad_norm"]),
        policy_kwargs={
            "net_arch": list(ppo_config["policy_net_arch"])
        },
        seed=int(config["seed"]),
        device="cpu",
        verbose=1,
    )
    model.learn(total_timesteps=int(config["total_timesteps"]))

    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "ppo_model"
    model.save(model_path)
    validation_results, action_counts = evaluate_model_on_episodes(
        model, validation_env
    )
    validation_path = output / "validation_episode_results.json"
    validation_path.write_text(
        json.dumps(
            validation_results,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    scores = np.asarray(
        [row["terminal_score"] for row in validation_results],
        dtype=np.float64,
    )
    run_manifest = {
        "status": "cpu_smoke_completed",
        "alpha_claim": False,
        "test_split_evaluated": False,
        "config": config,
        "config_sha256": sha256_file(args.config),
        "dataset_sha256": sha256_file(args.dataset),
        "model_path": str(model_path.with_suffix(".zip")),
        "validation_results_path": str(validation_path),
        "validation": {
            "episodes": len(validation_results),
            "mean_terminal_score": float(scores.mean()),
            "all_finite": bool(np.isfinite(scores).all()),
            "action_counts": action_counts,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            run_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        f"CPU smoke complete; validation mean score="
        f"{scores.mean():+.6f}; wrote {model_path.with_suffix('.zip')} "
        f"and {manifest_path}"
    )


if __name__ == "__main__":
    main()
