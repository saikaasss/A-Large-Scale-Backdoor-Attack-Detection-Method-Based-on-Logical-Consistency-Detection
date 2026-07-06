from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rule_features import extract_step_rule_features


LABEL2ID = {
    "baseline": 0,
    "epistemic": 1,
    "process": 2,
    "shortcut": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True, help="原始 stage3_train.jsonl 路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出新训练数据集路径")
    parser.add_argument("--stage2_model_path", type=str, required=True, help="已训练 stage2 模型目录")

    parser.add_argument("--max_length", type=int, default=512, help="stage2 tokenizer 最大长度")
    parser.add_argument("--batch_size", type=int, default=16, help="stage2 推理 batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda / cpu",
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
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"读取失败: {path}, line={line_idx}, err={e}") from e
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_steps(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(s).strip() for s in x if str(s).strip()]
    s = str(x).strip()
    return [s] if s else []


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


def build_stage2_text(
    premise: str,
    question: str,
    previous_steps: List[str],
    current_step: str,
) -> str:
    previous_steps_text = normalize_previous_steps(previous_steps)

    text = (
        f"Premise: {premise.strip()}\n\n"
        f"Question: {question.strip()}\n\n"
        f"Previous steps: {previous_steps_text}\n\n"
        f"Current step: {current_step.strip()}"
    )
    return text


def flatten_rows(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat_rows: List[Dict[str, Any]] = []

    for row_idx, row in enumerate(raw_rows):
        premise = str(row.get("premise", "")).strip()
        question = str(row.get("question", "")).strip()
        conclusion = str(row.get("conclusion", "")).strip()
        reasoning_steps = normalize_steps(row.get("reasoning_steps", []))

        for i, current_step in enumerate(reasoning_steps):
            previous_steps = reasoning_steps[:i]

            flat_rows.append(
                {
                    "row_idx": row_idx,
                    "sample_id": str(row.get("sample_id", "")).strip(),
                    "example_id": str(row.get("example_id", "")).strip(),
                    "condition": str(row.get("condition", "")).strip(),
                    "label": str(row.get("label", "")).strip(),
                    "premise": premise,
                    "question": question,
                    "conclusion": conclusion,
                    "step_id": i + 1,
                    "previous_steps": previous_steps,
                    "current_step": current_step,
                    "text": build_stage2_text(
                        premise=premise,
                        question=question,
                        previous_steps=previous_steps,
                        current_step=current_step,
                    ),
                }
            )

    return flat_rows


@torch.no_grad()
def predict_stage2(
    flat_rows: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    max_length: int,
    batch_size: int,
    device: str,
) -> List[Dict[str, Any]]:
    model.eval()
    outputs: List[Dict[str, Any]] = []

    for start in tqdm(range(0, len(flat_rows), batch_size), desc="Stage2 predicting"):
        batch = flat_rows[start:start + batch_size]
        texts = [x["text"] for x in batch]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)

        pred_ids = torch.argmax(probs, dim=-1).tolist()
        probs_list = probs.cpu().tolist()
        logits_list = logits.cpu().tolist()

        for item, pred_id, prob_vec, logit_vec in zip(batch, pred_ids, probs_list, logits_list):
            outputs.append(
                {
                    **item,
                    "stage2_pred_label": ID2LABEL[int(pred_id)],
                    "stage2_pred_id": int(pred_id),
                    "stage2_logits": {
                        "baseline": float(logit_vec[LABEL2ID["baseline"]]),
                        "epistemic": float(logit_vec[LABEL2ID["epistemic"]]),
                        "process": float(logit_vec[LABEL2ID["process"]]),
                        "shortcut": float(logit_vec[LABEL2ID["shortcut"]]),
                    },
                    "stage2_probs": {
                        "baseline": float(prob_vec[LABEL2ID["baseline"]]),
                        "epistemic": float(prob_vec[LABEL2ID["epistemic"]]),
                        "process": float(prob_vec[LABEL2ID["process"]]),
                        "shortcut": float(prob_vec[LABEL2ID["shortcut"]]),
                    },
                }
            )

    return outputs


def add_rule_features(step_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enhanced_rows: List[Dict[str, Any]] = []

    for row in step_rows:
        rule_feats = extract_step_rule_features(
            premise=row["premise"],
            question=row["question"],
            previous_steps=row["previous_steps"],
            current_step=row["current_step"],
            conclusion=row["conclusion"],
        )

        out = dict(row)
        out["rule_features"] = rule_feats
        enhanced_rows.append(out)

    return enhanced_rows


def build_model_input_text(
    premise: str,
    question: str,
    conclusion: str,
    enhanced_steps: List[Dict[str, Any]],
) -> str:
    blocks: List[str] = []
    blocks.append(f"Premise: {premise}")
    blocks.append(f"Question: {question}")

    for step in enhanced_steps:
        probs = step["stage2_probs"]
        rule = step["rule_features"]

        block = (
            f"Step {step['step_id']}:\n"
            f"Current step: {step['current_step']}\n"
            f"Pred label: {step['stage2_pred_label']}\n"
            f"Probs: "
            f"baseline={probs['baseline']:.4f}, "
            f"epistemic={probs['epistemic']:.4f}, "
            f"process={probs['process']:.4f}, "
            f"shortcut={probs['shortcut']:.4f}\n"
            f"Rules: "
            f"support={rule['rule_support_score']:.4f}, "
            f"contradiction={rule['rule_contradiction_score']:.4f}, "
            f"novelty={rule['rule_novelty_score']:.4f}, "
            f"premise_missing={rule['rule_premise_missing_score']:.4f}, "
            f"local_contradiction={rule['rule_local_contradiction_score']:.4f}, "
            f"new_claim={rule['rule_new_claim_score']:.4f}"
        )
        blocks.append(block)

    if conclusion:
        blocks.append(f"Conclusion: {conclusion}")

    return "\n\n".join(blocks)


def merge_to_new_dataset(
    raw_rows: List[Dict[str, Any]],
    enhanced_step_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in enhanced_step_rows:
        grouped.setdefault(row["row_idx"], []).append(row)

    new_rows: List[Dict[str, Any]] = []

    for row_idx, raw in enumerate(raw_rows):
        premise = str(raw.get("premise", "")).strip()
        question = str(raw.get("question", "")).strip()
        conclusion = str(raw.get("conclusion", "")).strip()
        reasoning_steps = normalize_steps(raw.get("reasoning_steps", []))

        steps = grouped.get(row_idx, [])
        steps = sorted(steps, key=lambda x: x["step_id"])

        enhanced_steps: List[Dict[str, Any]] = []
        for s in steps:
            enhanced_steps.append(
                {
                    "step_id": s["step_id"],
                    "previous_steps": s["previous_steps"],
                    "current_step": s["current_step"],
                    "stage2_pred_label": s["stage2_pred_label"],
                    "stage2_pred_id": s["stage2_pred_id"],
                    "stage2_logits": s["stage2_logits"],
                    "stage2_probs": s["stage2_probs"],
                    "rule_features": s["rule_features"],
                }
            )

        model_input_text = build_model_input_text(
            premise=premise,
            question=question,
            conclusion=conclusion,
            enhanced_steps=enhanced_steps,
        )

        out = {
            "sample_id": str(raw.get("sample_id", "")).strip(),
            "example_id": str(raw.get("example_id", "")).strip(),
            "condition": str(raw.get("condition", "")).strip(),
            "label": str(raw.get("label", "")).strip(),

            "premise": premise,
            "question": question,
            "conclusion": conclusion,
            "reasoning_steps": reasoning_steps,

            "enhanced_steps": enhanced_steps,
            "model_input_text": model_input_text,
        }
        new_rows.append(out)

    return new_rows


def main() -> int:
    args = parse_args()

    print("========== 读取原始 stage3 数据 ==========")
    raw_rows = read_jsonl(args.input_file)
    print(f"raw rows = {len(raw_rows)}")

    print("========== 展开为 step 级样本 ==========")
    flat_rows = flatten_rows(raw_rows)
    print(f"step rows = {len(flat_rows)}")

    print("========== 加载 stage2 模型 ==========")
    tokenizer = AutoTokenizer.from_pretrained(args.stage2_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.stage2_model_path)
    model.to(args.device)

    print("========== stage2 预测 ==========")
    pred_rows = predict_stage2(
        flat_rows=flat_rows,
        tokenizer=tokenizer,
        model=model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
    )

    print("========== 提取规则特征 ==========")
    enhanced_step_rows = add_rule_features(pred_rows)

    print("========== 构建新训练数据集 ==========")
    new_rows = merge_to_new_dataset(raw_rows, enhanced_step_rows)

    print("========== 保存 ==========")
    write_jsonl(args.output_file, new_rows)

    print(f"已保存到: {args.output_file}")
    print(f"新数据集样本数: {len(new_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

'''
python stage_fusion_dataset.py \
  --input_file ./data/stage3_train.jsonl \
  --output_file ./data/stage4_fusion.jsonl \
  --stage2_model_path ./outputs_stage2
'''