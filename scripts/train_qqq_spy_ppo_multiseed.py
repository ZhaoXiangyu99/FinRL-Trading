#!/usr/bin/env python3
"""Formal multi-seed PPO runner. Test data remains sealed by construction."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environments.gym_qqq_spy_cash import QQQSpyCashGymEnv
from src.environments.qqq_spy_cash_env import RewardConfig
from src.evaluation.trained_policy import evaluate_model_on_episodes
from src.training.data_pipeline import prepare_training_splits


DEFAULT_CONFIG = Path("configs/qqq_spy_agent/ppo_gpu_v1.json")
DEFAULT_DATASET = Path(
    "data/market/longbridge/qqq_spy_rl_training_dataset_v1.csv"
)
DEFAULT_METADATA = DEFAULT_DATASET.with_suffix(
    DEFAULT_DATASET.suffix + ".metadata.json"
)
DEFAULT_BASELINES = Path(
    "artifacts/qqq_spy_agent/baselines_v1/constant_policy_summary.csv"
)
DEFAULT_OUTPUT = Path("artifacts/qqq_spy_agent/ppo_multiseed_v1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_episodes(frame: pd.DataFrame) -> list[pd.DataFrame]:
    return [
        episode.reset_index(drop=True)
        for _, episode in frame.groupby("episode_id", sort=True)
    ]


def make_environment(
    episode_frames: list[pd.DataFrame],
    feature_columns: list[str],
    transaction_cost_rate: float,
    reward_config: RewardConfig,
) -> QQQSpyCashGymEnv:
    return QQQSpyCashGymEnv(
        episodes=episode_frames,
        feature_columns=feature_columns,
        transaction_cost_rate=transaction_cost_rate,
        reward_config=reward_config,
        episode_selection="random",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen multi-seed PPO training on train/validation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-metadata", type=Path, default=DEFAULT_METADATA
    )
    parser.add_argument(
        "--baseline-summary", type=Path, default=DEFAULT_BASELINES
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, splits, and environment without training.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        help="Explicitly override the frozen device.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    metadata = json.loads(
        args.dataset_metadata.read_text(encoding="utf-8")
    )
    feature_columns = metadata["feature_columns"]
    dataset = pd.read_csv(
        args.dataset,
        parse_dates=["feature_date", "target_date"],
    )
    splits, _ = prepare_training_splits(
        dataset, feature_columns=feature_columns
    )
    train_episodes = split_episodes(splits["train"])
    validation_episodes = split_episodes(splits["validation"])
    reward_config = RewardConfig(**config["reward"])
    transaction_cost = float(config["transaction_cost_rate"])

    check_environment = make_environment(
        train_episodes,
        feature_columns,
        transaction_cost,
        reward_config,
    )
    check_env(check_environment, warn=True, skip_render_check=True)
    if args.dry_run:
        print(
            f"Dry run OK: train={len(train_episodes)} episodes, "
            f"validation={len(validation_episodes)} episodes, "
            f"features={len(feature_columns)}, actions=15; test not loaded "
            f"into any environment"
        )
        return

    device = args.device or config["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; stop before training"
        )

    baseline_summary = pd.read_csv(args.baseline_summary)
    validation_baselines = baseline_summary.loc[
        baseline_summary["split"] == "validation"
    ]
    best_baseline_score = float(
        validation_baselines["mean_terminal_score"].max()
    )
    ppo_config = config["ppo"]
    run_summaries = []
    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)

    for seed in config["seeds"]:
        env_fns = [
            (
                lambda: make_environment(
                    train_episodes,
                    feature_columns,
                    transaction_cost,
                    reward_config,
                )
            )
            for _ in range(int(config["n_envs"]))
        ]
        vector_env = DummyVecEnv(env_fns)
        model = PPO(
            "MlpPolicy",
            vector_env,
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
            seed=int(seed),
            device=device,
            verbose=1,
        )
        model.learn(
            total_timesteps=int(config["total_timesteps_per_seed"])
        )
        model_path = output / f"ppo_seed_{seed}"
        model.save(model_path)

        validation_env = QQQSpyCashGymEnv(
            episodes=validation_episodes,
            feature_columns=feature_columns,
            transaction_cost_rate=transaction_cost,
            reward_config=reward_config,
            episode_selection="sequential",
        )
        results, action_counts = evaluate_model_on_episodes(
            model, validation_env
        )
        result_path = output / f"validation_seed_{seed}.json"
        result_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        scores = np.asarray(
            [row["terminal_score"] for row in results], dtype=np.float64
        )
        drawdowns = np.asarray(
            [row["max_drawdown"] for row in results], dtype=np.float64
        )
        turnovers = np.asarray(
            [row["cumulative_turnover"] for row in results],
            dtype=np.float64,
        )
        run_summaries.append(
            {
                "seed": int(seed),
                "model_path": str(model_path.with_suffix(".zip")),
                "validation_results": str(result_path),
                "mean_terminal_score": float(scores.mean()),
                "worst_half_year_drawdown": float(drawdowns.max()),
                "mean_half_year_turnover": float(turnovers.mean()),
                "all_finite": bool(
                    np.isfinite(scores).all()
                    and np.isfinite(drawdowns).all()
                    and np.isfinite(turnovers).all()
                ),
                "action_counts": action_counts,
            }
        )
        vector_env.close()

    best = max(
        run_summaries, key=lambda row: row["mean_terminal_score"]
    )
    gate = config["selection"]
    gates = {
        "finite": best["all_finite"],
        "beats_constant_policy_margin": (
            best["mean_terminal_score"]
            >= best_baseline_score
            + float(gate["minimum_margin_over_best_constant_policy"])
        ),
        "drawdown": (
            best["worst_half_year_drawdown"]
            <= float(gate["maximum_worst_half_year_drawdown"])
        ),
        "turnover": (
            best["mean_half_year_turnover"]
            <= float(gate["maximum_mean_half_year_turnover"])
        ),
        "test_split_sealed": True,
    }
    manifest = {
        "status": (
            "validation_challenger"
            if all(gates.values())
            else "validation_rejected"
        ),
        "champion_promoted": False,
        "test_split_evaluated": False,
        "config_sha256": sha256_file(args.config),
        "dataset_sha256": sha256_file(args.dataset),
        "best_constant_validation_score": best_baseline_score,
        "runs": run_summaries,
        "selected_run": best,
        "gates": gates,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device,
            "cuda_device": (
                torch.cuda.get_device_name(0) if device == "cuda" else None
            ),
        },
    }
    manifest_path = output / "selection_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote frozen selection manifest to {manifest_path}")


if __name__ == "__main__":
    main()
