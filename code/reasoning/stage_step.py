from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABEL2ID = {
    "baseline": 0,
    "epistemic": 1,
    "process": 2,
    "shortcut": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_file",
        type=str,
        default="./stage2_train.jsonl",
        help="训练集 jsonl 文件路径",
    )
    parser.add_argument(
        "--dev_file",
        type=str,
        default="./stage2_dev.jsonl",
        help="验证集 jsonl 文件路径",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/deberta-v3-base",
        help="Hugging Face 模型名或本地模型目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_stage2",
        help="输出目录",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="tokenizer 最大长度",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=4,
        help="训练轮数",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="学习率",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="weight decay",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="单卡训练 batch size",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="单卡验证 batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="warmup 比例",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="日志打印步数",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="验证策略",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["no", "steps", "epoch"],
        help="保存策略",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="最多保留几个 checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="开启 fp16",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="开启 bf16",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="开启 gradient checkpointing",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help='日志上报目标，如 "none" / "tensorboard"',
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="是否保存验证集预测结果",
    )

    return parser.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON 解析失败: {path}, 第 {line_idx} 行, 错误: {e}") from e
    return rows


def normalize_previous_steps(previous_steps: Any) -> str:
    if previous_steps is None:
        return "None"

    if isinstance(previous_steps, list):
        cleaned = []
        for x in previous_steps:
            s = str(x).strip()
            if s:
                cleaned.append(s)
        return " ".join(cleaned) if cleaned else "None"

    s = str(previous_steps).strip()
    return s if s else "None"


def build_text(example: Dict[str, Any]) -> str:
    premise = str(example.get("premise", "")).strip()
    question = str(example.get("question", "")).strip()
    previous_steps = normalize_previous_steps(example.get("previous_steps", None))
    current_step = str(example.get("current_step", "")).strip()

    text = (
        f"Premise: {premise}\n\n"
        f"Question: {question}\n\n"
        f"Previous steps: {previous_steps}\n\n"
        f"Current step: {current_step}"
    )
    return text


def convert_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        label = str(row.get("label", "")).strip()
        if label not in LABEL2ID:
            raise ValueError(
                f"第 {i} 条样本 label 不合法: {label!r}，允许值: {list(LABEL2ID.keys())}"
            )

        converted.append(
            {
                "text": build_text(row),
                "label": LABEL2ID[label],
                "label_name": label,
                "sample_id": str(row.get("sample_id", "")).strip(),
                "example_id": str(row.get("example_id", "")).strip(),
                "condition": str(row.get("condition", "")).strip(),
                "step_id": row.get("step_id", None),
            }
        )
    return converted


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        labels, preds, labels=list(ID2LABEL.keys()), average=None, zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
    }

    for i, label_name in ID2LABEL.items():
        metrics[f"precision_{label_name}"] = per_class_p[i]
        metrics[f"recall_{label_name}"] = per_class_r[i]
        metrics[f"f1_{label_name}"] = per_class_f1[i]

    return metrics


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

    keep_columns = [
        "input_ids",
        "attention_mask",
        "label",
    ]
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
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print("========== 训练参数 ==========")
    num_update_steps_per_epoch = max(
        1,
        len(train_dataset) // (
                args.per_device_train_batch_size * max(1, args.gradient_accumulation_steps)
        )
    )
    total_train_steps = int(num_update_steps_per_epoch * args.num_train_epochs)
    warmup_steps = int(total_train_steps * args.warmup_ratio)

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
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
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
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    print("========== 开始验证 ==========")
    eval_metrics = trainer.evaluate(dev_dataset)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    if args.save_predictions:
        print("========== 保存验证集预测 ==========")
        preds_output = trainer.predict(dev_dataset)
        logits = preds_output.predictions
        pred_ids = np.argmax(logits, axis=-1)
        gold_ids = preds_output.label_ids

        # softmax 概率
        probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()

        pred_path = Path(args.output_dir) / "dev_predictions.jsonl"
        with open(pred_path, "w", encoding="utf-8") as f:
            raw_dev_rows = convert_rows(read_jsonl(args.dev_file))
            for i, row in enumerate(raw_dev_rows):
                out = {
                    "sample_id": row["sample_id"],
                    "example_id": row["example_id"],
                    "condition": row["condition"],
                    "step_id": row["step_id"],

                    "gold_label": ID2LABEL[int(gold_ids[i])],
                    "pred_label": ID2LABEL[int(pred_ids[i])],
                    "correct": int(gold_ids[i] == pred_ids[i]),

                    # logits
                    "logit_baseline": float(logits[i][LABEL2ID["baseline"]]),
                    "logit_epistemic": float(logits[i][LABEL2ID["epistemic"]]),
                    "logit_process": float(logits[i][LABEL2ID["process"]]),
                    "logit_shortcut": float(logits[i][LABEL2ID["shortcut"]]),

                    # probabilities
                    "p_baseline": float(probs[i][LABEL2ID["baseline"]]),
                    "p_epistemic": float(probs[i][LABEL2ID["epistemic"]]),
                    "p_process": float(probs[i][LABEL2ID["process"]]),
                    "p_shortcut": float(probs[i][LABEL2ID["shortcut"]]),
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        print(f"预测结果已保存到: {pred_path}")

    print("========== 训练完成 ==========")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

'''
nohup python stage_2.py \
  --train_file ./data/stage2_train.jsonl \
  --dev_file ./data/stage2_dev.jsonl \
  --model_name_or_path ./roberta-base \
  --output_dir ./outputs_stage2 \
  --num_train_epochs 4 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --max_length 512 \
  --fp16 \
  --save_predictions\
  > train_stage_2.log 2>&1 &
'''