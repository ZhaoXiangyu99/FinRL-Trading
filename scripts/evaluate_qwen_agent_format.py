#!/usr/bin/env python3
"""Generate validation decisions and score the strict action JSON contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentic.contracts import parse_and_validate_action_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/qqq_spy_agent/qwen3_8b_qlora_sft_v1.json"
        ),
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_validation(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    import random

    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    validation_path = Path(config["data"]["validation"])
    if not validation_path.is_absolute():
        validation_path = PROJECT_ROOT / validation_path
    rows = _load_validation(validation_path)
    selected = random.Random(args.seed).sample(
        rows, min(args.samples, len(rows))
    )

    model_path = config["model"]["local_path"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()

    details = []
    valid_count = 0
    teacher_agreement_count = 0
    for row in selected:
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            prompt_text = tokenizer.apply_chat_template(
                row["prompt"],
                enable_thinking=False,
                **template_kwargs,
            )
        except TypeError:
            prompt_text = tokenizer.apply_chat_template(
                row["prompt"],
                **template_kwargs,
            )
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        error = None
        predicted_action = None
        try:
            proposal = parse_and_validate_action_json(generated)
            predicted_action = proposal["action_id"]
            valid_count += 1
        except ValueError as exc:
            error = str(exc)
        expected = json.loads(row["completion"][0]["content"])["action_id"]
        if predicted_action == expected:
            teacher_agreement_count += 1
        details.append(
            {
                "sample_id": row["sample_id"],
                "expected_teacher_action_id": expected,
                "predicted_action_id": predicted_action,
                "schema_valid": error is None,
                "error": error,
                "generated": generated,
            }
        )

    count = len(selected)
    report = {
        "evaluation": "validation_format_only_not_return_performance",
        "adapter": str(args.adapter),
        "samples": count,
        "seed": args.seed,
        "schema_valid_count": valid_count,
        "schema_valid_rate": valid_count / count,
        "teacher_agreement_count": teacher_agreement_count,
        "teacher_agreement_rate": teacher_agreement_count / count,
        "test_samples_used": 0,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "details"},
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
