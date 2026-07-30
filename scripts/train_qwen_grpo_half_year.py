#!/usr/bin/env python3
"""Train the Qwen adapter with counterfactual half-year GRPO rewards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentic.grpo_rewards import (
    half_year_outcome_reward,
    strict_format_reward,
)
from src.agentic.half_year_rewards import REWARD_TABLE_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/qqq_spy_agent/qwen3_8b_grpo_half_year_v1.json"
        ),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    return parser.parse_args()


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_rows(path: Path, expected_split: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("reward_table_version") != REWARD_TABLE_VERSION:
                raise ValueError(
                    f"{path}:{line_number}: unexpected reward version"
                )
            if row.get("split") != expected_split:
                raise ValueError(
                    f"{path}:{line_number}: unexpected split"
                )
            if len(row.get("action_rewards", [])) != 15:
                raise ValueError(
                    f"{path}:{line_number}: expected 15 action rewards"
                )
            serialized_prompt = json.dumps(
                row["prompt"], ensure_ascii=False
            )
            forbidden = (
                "action_rewards",
                "terminal_score",
                "overnight_return",
                "intraday_return",
                "close_to_close_return",
            )
            if any(value in serialized_prompt for value in forbidden):
                raise ValueError(
                    f"{path}:{line_number}: future outcome leaked to prompt"
                )
            rows.append(
                {
                    "prompt": row["prompt"],
                    "sample_id": row["sample_id"],
                    "action_rewards": row["action_rewards"],
                }
            )
    if not rows:
        raise ValueError(f"empty reward dataset: {path}")
    return rows


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GRPO")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the GPU must support bfloat16")

    metadata = json.loads(
        _resolve(config["data"]["metadata"]).read_text(encoding="utf-8")
    )
    controls = metadata["controls"]
    if controls["test_rows_written"]:
        raise RuntimeError("refusing reward data containing test rows")
    if controls["test_used_for_training_or_selection"]:
        raise RuntimeError("refusing reward data that used test")
    if controls["future_returns_in_prompt"]:
        raise RuntimeError("refusing prompts containing future returns")
    if not controls["future_returns_in_reward"]:
        raise RuntimeError("outcome RL requires realized reward data")

    train_rows = _load_rows(
        _resolve(config["data"]["train"]), "train"
    )
    seed = int(config["training"]["seed"])
    random.Random(seed).shuffle(train_rows)
    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max-train-samples must be positive")
        train_rows = train_rows[: args.max_train_samples]

    model_path = config["model"]["local_path"]
    adapter_path = Path(config["model"]["sft_adapter"])
    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"base model missing: {model_path}")
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"SFT adapter missing: {adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = config["quantization"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=bool(quant["load_in_4bit"]),
        bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=bool(
            quant["bnb_4bit_use_double_quant"]
        ),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=True,
    )

    training = config["training"]
    max_steps = (
        int(args.max_steps)
        if args.max_steps is not None
        else int(training["max_steps"])
    )
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    output_dir = args.output_dir or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    num_generations = (
        int(args.num_generations)
        if args.num_generations is not None
        else int(training["num_generations"])
    )
    temperature = (
        float(args.temperature)
        if args.temperature is not None
        else float(training["temperature"])
    )
    if num_generations < 2:
        raise ValueError("num_generations must be at least 2")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        run_name=config["run_name"],
        seed=seed,
        data_seed=seed,
        max_steps=max_steps,
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=int(
            training["gradient_accumulation_steps"]
        ),
        learning_rate=float(training["learning_rate"]),
        warmup_steps=int(training["warmup_steps"]),
        lr_scheduler_type=training["lr_scheduler_type"],
        logging_steps=int(training["logging_steps"]),
        save_strategy="steps",
        save_steps=int(training["save_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        gradient_checkpointing=bool(
            training["gradient_checkpointing"]
        ),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bool(training["bf16"]),
        tf32=bool(training["tf32"]),
        optim=training["optim"],
        report_to=training["report_to"],
        num_generations=num_generations,
        generation_batch_size=num_generations,
        max_completion_length=int(
            training["max_completion_length"]
        ),
        temperature=temperature,
        generation_kwargs={"top_p": float(training["top_p"])},
        chat_template_kwargs={"enable_thinking": False},
        beta=float(training["beta"]),
        loss_type=training["loss_type"],
        mask_truncated_completions=bool(
            training["mask_truncated_completions"]
        ),
        reward_weights=[
            float(config["reward_weights"]["half_year_outcome"]),
            float(config["reward_weights"]["strict_format"]),
        ],
        remove_unused_columns=False,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            half_year_outcome_reward,
            strict_format_reward,
        ],
        args=grpo_config,
        train_dataset=Dataset.from_list(train_rows),
        processing_class=tokenizer,
    )
    trainer.model.print_trainable_parameters()
    result = trainer.train()
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    tokenizer.save_pretrained(str(final_adapter))
    metrics = dict(result.metrics)
    metrics.update(
        {
            "algorithm": "GRPO",
            "reward_contract": REWARD_TABLE_VERSION,
            "base_model_id": config["model"]["id"],
            "sft_adapter": str(adapter_path),
            "train_samples_available": len(train_rows),
            "test_samples_used": 0,
            "cuda_device": torch.cuda.get_device_name(0),
            "max_memory_allocated_gib": round(
                torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
        }
    )
    (output_dir / "run_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
