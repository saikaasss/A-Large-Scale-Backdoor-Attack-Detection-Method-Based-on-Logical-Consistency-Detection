from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABEL2ID = {
    "Incorrect": 0,
    "Correct": 1,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", type=str, default="./stage4_fusion_train.jsonl")
    parser.add_argument("--dev_file", type=str, default="./stage4_fusion_dev.jsonl")
    parser.add_argument("--model_name_or_path", type=str, default="FacebookAI/roberta-base")
    parser.add_argument("--output_dir", type=str, default="./outputs_stage4_fusion")

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=50)

    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
    )
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--save_predictions", action="store_true")

    return parser.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败: {path}, 第 {line_idx} 行, 错误: {e}") from e
    return rows


def normalize_reasoning_steps(reasoning_steps: Any) -> str:
    if reasoning_steps is None:
        return "None"

    if isinstance(reasoning_steps, list):
        cleaned: List[str] = []
        for i, x in enumerate(reasoning_steps, start=1):
            s = str(x).strip()
            if s:
                cleaned.append(f"Step {i}: {s}")
        return "\n".join(cleaned) if cleaned else "None"

    s = str(reasoning_steps).strip()
    return s if s else "None"


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def build_text_from_enhanced_steps(example: Dict[str, Any]) -> str:
    premise = str(example.get("premise", "")).strip()
    question = str(example.get("question", "")).strip()
    conclusion = str(example.get("conclusion", "")).strip()
    enhanced_steps = example.get("enhanced_steps", [])

    blocks: List[str] = []
    blocks.append(f"Premise: {premise}")
    blocks.append(f"Question: {question}")

    if not isinstance(enhanced_steps, list) or len(enhanced_steps) == 0:
        blocks.append("Enhanced steps: None")
    else:
        for step in enhanced_steps:
            step_id = step.get("step_id", "")
            current_step = str(step.get("current_step", "")).strip()
            pred_label = str(step.get("stage2_pred_label", "")).strip()

            probs = step.get("stage2_probs", {}) or {}
            rules = step.get("rule_features", {}) or {}

            block = (
                f"Step {step_id}:\n"
                f"Current step: {current_step}\n"
                f"Pred label: {pred_label}\n"
                f"Probs: "
                f"baseline={safe_float(probs.get('baseline', 0.0)):.4f}, "
                f"epistemic={safe_float(probs.get('epistemic', 0.0)):.4f}, "
                f"process={safe_float(probs.get('process', 0.0)):.4f}, "
                f"shortcut={safe_float(probs.get('shortcut', 0.0)):.4f}\n"
                f"Rules: "
                f"support={safe_float(rules.get('rule_support_score', 0.0)):.4f}, "
                f"contradiction={safe_float(rules.get('rule_contradiction_score', 0.0)):.4f}, "
                f"novelty={safe_float(rules.get('rule_novelty_score', 0.0)):.4f}, "
                f"premise_missing={safe_float(rules.get('rule_premise_missing_score', 0.0)):.4f}, "
                f"local_contradiction={safe_float(rules.get('rule_local_contradiction_score', 0.0)):.4f}, "
                f"new_claim={safe_float(rules.get('rule_new_claim_score', 0.0)):.4f}"
            )
            blocks.append(block)

    if conclusion:
        blocks.append(f"Conclusion: {conclusion}")

    return "\n\n".join(blocks)


def build_text_from_old_fields(example: Dict[str, Any]) -> str:
    premise = str(example.get("premise", "")).strip()
    question = str(example.get("question", "")).strip()
    reasoning_steps = normalize_reasoning_steps(example.get("reasoning_steps", None))
    conclusion = str(example.get("conclusion", "")).strip()

    text = (
        f"Premise: {premise}\n\n"
        f"Question: {question}\n\n"
        f"Reasoning steps:\n{reasoning_steps}\n\n"
        f"Conclusion: {conclusion}"
    )
    return text


def build_text(example: Dict[str, Any]) -> str:
    """
    数据输入优先级：
    1. model_input_text
    2. enhanced_steps
    3. 旧字段 premise/question/reasoning_steps/conclusion
    """
    model_input_text = str(example.get("model_input_text", "")).strip()
    if model_input_text:
        return model_input_text

    enhanced_steps = example.get("enhanced_steps", None)
    if isinstance(enhanced_steps, list) and len(enhanced_steps) > 0:
        return build_text_from_enhanced_steps(example)

    return build_text_from_old_fields(example)


def map_label(row: Dict[str, Any]) -> int:
    src_label = str(row.get("label", "")).strip()
    if src_label not in LABEL2ID:
        raise ValueError(
            f"原始 label 不合法: {src_label!r}，允许值: {sorted(LABEL2ID.keys())}"
        )
    return LABEL2ID[src_label]


def convert_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        label_id = map_label(row)
        text = build_text(row)

        if not text.strip():
            raise ValueError(f"第 {i} 条样本转换后 text 为空，请检查输入字段。")

        converted.append(
            {
                "text": text,
                "label": label_id,
                "label_name": ID2LABEL[label_id],
                "raw_label": str(row.get("label", "")).strip(),
                "sample_id": str(row.get("sample_id", "")).strip(),
                "example_id": str(row.get("example_id", "")).strip(),
                "condition": str(row.get("condition", "")).strip(),
            }
        )
    return converted


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0, pos_label=1
    )
    p_each, r_each, f1_each, support_each = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], average=None, zero_division=0
    )

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "accuracy": acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "binary_precision_correct": p_bin,
        "binary_recall_correct": r_bin,
        "binary_f1_correct": f1_bin,
        "precision_incorrect": p_each[0],
        "recall_incorrect": r_each[0],
        "f1_incorrect": f1_each[0],
        "support_incorrect": support_each[0],
        "precision_correct": p_each[1],
        "recall_correct": r_each[1],
        "f1_correct": f1_each[1],
        "support_correct": support_each[1],
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("========== 读取数据 ==========")
    train_rows = read_jsonl(args.train_file)
    dev_rows = read_jsonl(args.dev_file)
    print(f"train size = {len(train_rows)}")
    print(f"dev size   = {len(dev_rows)}")

    print("========== 转换数据 ==========")
    train_data = convert_rows(train_rows)
    dev_data = convert_rows(dev_rows)

    if len(train_data) > 0:
        print("========== 训练样本示例 ==========")
        print(train_data[0]["text"][:1000])

    train_dataset = Dataset.from_list(train_data)
    dev_dataset = Dataset.from_list(dev_data)

    print("========== 加载 tokenizer ==========")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def tokenize_fn(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
        )

    print("========== tokenizer 编码 ==========")
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    dev_dataset = dev_dataset.map(tokenize_fn, batched=True)

    keep_columns = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in train_dataset.column_names:
        keep_columns.append("token_type_ids")

    train_dataset = train_dataset.remove_columns(
        [c for c in train_dataset.column_names if c not in keep_columns]
    )
    dev_dataset = dev_dataset.remove_columns(
        [c for c in dev_dataset.column_names if c not in keep_columns]
    )

    print("========== 加载模型 ==========")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    num_update_steps_per_epoch = max(
        1,
        len(train_dataset) // (
            args.per_device_train_batch_size * max(1, args.gradient_accumulation_steps)
        )
    )
    total_train_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
    warmup_steps = int(total_train_steps * args.warmup_ratio)

    print("========== 训练参数 ==========")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=(args.eval_strategy != "no"),
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=(args.eval_strategy != "no" and args.save_strategy != "no"),
        metric_for_best_model="binary_f1_correct",
        greater_is_better=True,
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        report_to=args.report_to,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset if args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if args.eval_strategy != "no" else None,
    )

    print("========== 开始训练 ==========")
    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("========== 保存训练日志 ==========")
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    if args.eval_strategy != "no":
        print("========== 开始验证 ==========")
        eval_metrics = trainer.evaluate(dev_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    if args.save_predictions:
        print("========== 保存验证集预测 ==========")
        pred_output = trainer.predict(dev_dataset)
        logits = pred_output.predictions
        pred_ids = np.argmax(logits, axis=-1)
        gold_ids = pred_output.label_ids

        probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()

        pred_path = Path(args.output_dir) / "dev_predictions_binary.jsonl"
        raw_dev_rows = convert_rows(read_jsonl(args.dev_file))

        with open(pred_path, "w", encoding="utf-8") as f:
            for i, row in enumerate(raw_dev_rows):
                out = {
                    "sample_id": row["sample_id"],
                    "example_id": row["example_id"],
                    "condition": row["condition"],
                    "raw_label": row["raw_label"],

                    "gold_label": ID2LABEL[int(gold_ids[i])],
                    "pred_label": ID2LABEL[int(pred_ids[i])],
                    "correct": int(gold_ids[i] == pred_ids[i]),

                    "logit_incorrect": float(logits[i][LABEL2ID["Incorrect"]]),
                    "logit_correct": float(logits[i][LABEL2ID["Correct"]]),

                    "p_incorrect": float(probs[i][LABEL2ID["Incorrect"]]),
                    "p_correct": float(probs[i][LABEL2ID["Correct"]]),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        print(f"预测结果已保存到: {pred_path}")

    print("========== 训练完成 ==========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

'''
python stage_fusion.py \
  --train_file ./data/stage4_fusion_train.jsonl \
  --dev_file ./data/stage4_fusion_dev.jsonl \
  --model_name_or_path ./roberta-base \
  --output_dir ./outputs_stage4_fusion \
  --num_train_epochs 4 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --max_length 512 \
  --save_predictions
'''