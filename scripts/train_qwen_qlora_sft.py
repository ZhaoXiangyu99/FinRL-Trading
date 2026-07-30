#!/usr/bin/env python3
"""Run 4-bit QLoRA SFT for the constrained QQQ/SPY policy interface."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentic.contracts import validate_trajectory_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/qqq_spy_agent/qwen3_8b_qlora_sft_v1.json"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override for a bounded smoke run.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Deterministically cap train samples for a smoke run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output location.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            try:
                validate_trajectory_record(row)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: {exc}"
                ) from exc
            rows.append(row)
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    return rows


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))

    # Heavy imports stay inside main so --help and contract tests do not need
    # the GPU-only stack.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QLoRA entrypoint")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected GPU must support bfloat16")

    train_path = _resolve(PROJECT_ROOT, config["data"]["train"])
    validation_path = _resolve(PROJECT_ROOT, config["data"]["validation"])
    metadata_path = _resolve(PROJECT_ROOT, config["data"]["metadata"])
    train_rows = _load_jsonl(train_path)
    validation_rows = _load_jsonl(validation_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["controls"]["test_records_written"]:
        raise RuntimeError("refusing data metadata that includes test records")
    if metadata["controls"]["test_used_for_fit_or_selection"]:
        raise RuntimeError("refusing data metadata that uses the test split")

    seed = int(config["training"]["seed"])
    random.Random(seed).shuffle(train_rows)
    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max-train-samples must be positive")
        train_rows = train_rows[: args.max_train_samples]

    model_path = config["model"]["local_path"]
    if not Path(model_path).exists():
        model_path = config["model"]["id"]
    quant = config["quantization"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=bool(quant["load_in_4bit"]),
        bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=bool(
            quant["bnb_4bit_use_double_quant"]
        ),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=(
            config["model"]["revision"]
            if model_path == config["model"]["id"]
            else None
        ),
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=(
            config["model"]["revision"]
            if model_path == config["model"]["id"]
            else None
        ),
        quantization_config=quantization_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(
            config["training"]["gradient_checkpointing"]
        ),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    lora = config["lora"]
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["lora_alpha"]),
        lora_dropout=float(lora["lora_dropout"]),
        target_modules=lora["target_modules"],
        bias=lora["bias"],
        task_type="CAUSAL_LM",
    )

    output_dir = args.output_dir or Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    sft_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "run_name": config["run_name"],
        "seed": seed,
        "data_seed": seed,
        "max_length": int(training["max_length"]),
        "num_train_epochs": float(training["num_train_epochs"]),
        "per_device_train_batch_size": int(
            training["per_device_train_batch_size"]
        ),
        "per_device_eval_batch_size": int(
            training["per_device_eval_batch_size"]
        ),
        "gradient_accumulation_steps": int(
            training["gradient_accumulation_steps"]
        ),
        "learning_rate": float(training["learning_rate"]),
        "warmup_ratio": float(training["warmup_ratio"]),
        "lr_scheduler_type": training["lr_scheduler_type"],
        "logging_steps": int(training["logging_steps"]),
        "eval_strategy": "steps",
        "eval_steps": int(training["eval_steps"]),
        "save_strategy": "steps",
        "save_steps": int(training["save_steps"]),
        "save_total_limit": int(training["save_total_limit"]),
        "gradient_checkpointing": bool(
            training["gradient_checkpointing"]
        ),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": bool(training["bf16"]),
        "tf32": bool(training["tf32"]),
        "packing": bool(training["packing"]),
        "optim": training["optim"],
        "report_to": training["report_to"],
        "completion_only_loss": True,
        "remove_unused_columns": False,
    }
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        sft_kwargs["max_steps"] = args.max_steps

    # TRL renamed evaluation_strategy to eval_strategy. Keep a narrow fallback
    # so the config remains usable with the supported 0.x range.
    sft_signature = inspect.signature(SFTConfig.__init__).parameters
    if "eval_strategy" not in sft_signature:
        sft_kwargs["evaluation_strategy"] = sft_kwargs.pop("eval_strategy")
    sft_config = SFTConfig(**sft_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(validation_rows),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))
    metrics = dict(train_result.metrics)
    metrics.update(
        {
            "base_model_id": config["model"]["id"],
            "base_model_revision": config["model"]["revision"],
            "train_samples": len(train_rows),
            "validation_samples": len(validation_rows),
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
    # Avoid tokenizer worker explosions on a single rented GPU.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
