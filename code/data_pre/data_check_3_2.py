from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =========================================================
# 路径配置：直接改这里
# =========================================================
BASE_TRAIN_FILE = "./data/stage3_train.jsonl"
BASE_DEV_FILE = "./data/stage3_dev.jsonl"

STAGE2_MODEL_DIR = "./outputs_stage2"   # 训练好的 stage_2 模型目录
STAGE3_MODEL_DIR = "./outputs_stage3"   # 训练好的 stage_3 模型目录

OUT_TRAIN_FILE = "./data/fusion_train.jsonl"
OUT_DEV_FILE = "./data/fusion_dev.jsonl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STAGE2_MAX_LENGTH = 512
STAGE3_MAX_LENGTH = 512


# =========================================================
# 标签定义
# =========================================================
STAGE2_LABELS = ["baseline", "epistemic", "process", "shortcut"]
STAGE2_LABEL2ID = {x: i for i, x in enumerate(STAGE2_LABELS)}
STAGE2_ID2LABEL = {i: x for x, i in STAGE2_LABEL2ID.items()}

FUSION_LABEL_MAP = {
    "Correct": "Correct",
    "Incorrect": "Incorrect",
}

CONDITION_TO_BINARY = {
    "baseline": "Correct",
    "epistemic": "Incorrect",
    "process": "Incorrect",
    "shortcut": "Incorrect",
}

MAJOR_ERROR_TYPE_ID = {
    "baseline": 0,
    "epistemic": 1,
    "process": 2,
    "shortcut": 3,
}


# =========================================================
# 通用函数
# =========================================================
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
                raise ValueError(f"读取 JSONL 失败: {path}, line={line_idx}, err={e}") from e
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def safe_len_text(x: Any) -> int:
    return len(safe_text(x))


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_x = np.exp(logits)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def get_reasoning_steps(row: Dict[str, Any]) -> List[str]:
    steps = row.get("reasoning_steps", [])
    if not isinstance(steps, list):
        return []
    out: List[str] = []
    for s in steps:
        s2 = safe_text(s)
        if s2:
            out.append(s2)
    return out


# =========================================================
# 与训练脚本对齐的文本构造
# =========================================================
def normalize_previous_steps(previous_steps: Any) -> str:
    if previous_steps is None:
        return "None"
    if isinstance(previous_steps, list):
        cleaned: List[str] = []
        for x in previous_steps:
            s = safe_text(x)
            if s:
                cleaned.append(s)
        return " ".join(cleaned) if cleaned else "None"
    s = safe_text(previous_steps)
    return s if s else "None"


def normalize_reasoning_steps(reasoning_steps: Any) -> str:
    if reasoning_steps is None:
        return "None"
    if isinstance(reasoning_steps, list):
        cleaned: List[str] = []
        for i, x in enumerate(reasoning_steps, start=1):
            s = safe_text(x)
            if s:
                cleaned.append(f"Step {i}: {s}")
        return "\n".join(cleaned) if cleaned else "None"
    s = safe_text(reasoning_steps)
    return s if s else "None"


def build_stage2_input(
    premise: str,
    question: str,
    previous_steps: List[str],
    current_step: str,
) -> str:
    prev_text = normalize_previous_steps(previous_steps)
    return (
        f"Premise: {premise}\n\n"
        f"Question: {question}\n\n"
        f"Previous steps: {prev_text}\n\n"
        f"Current step: {current_step}"
    )


def build_stage3_input(
    premise: str,
    question: str,
    reasoning_steps: List[str],
    conclusion: str,
) -> str:
    steps_text = normalize_reasoning_steps(reasoning_steps)
    return (
        f"Premise: {premise}\n\n"
        f"Question: {question}\n\n"
        f"Reasoning steps:\n{steps_text}\n\n"
        f"Conclusion: {conclusion}"
    )


# =========================================================
# 模型封装
# =========================================================
class Stage2Predictor:
    def __init__(self, model_dir: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(DEVICE)
        self.model.eval()
        self.max_length = max_length

    @torch.no_grad()
    def predict_one(self, text: str) -> Dict[str, Any]:
        batch = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = self.model(**batch)
        logits = outputs.logits[0].detach().cpu().numpy()
        probs = softmax_np(np.expand_dims(logits, axis=0))[0]
        pred_id = int(np.argmax(probs))

        return {
            "pred_label": STAGE2_ID2LABEL[pred_id],
            "logit_baseline": float(logits[STAGE2_LABEL2ID["baseline"]]),
            "logit_epistemic": float(logits[STAGE2_LABEL2ID["epistemic"]]),
            "logit_process": float(logits[STAGE2_LABEL2ID["process"]]),
            "logit_shortcut": float(logits[STAGE2_LABEL2ID["shortcut"]]),
            "p_baseline": float(probs[STAGE2_LABEL2ID["baseline"]]),
            "p_epistemic": float(probs[STAGE2_LABEL2ID["epistemic"]]),
            "p_process": float(probs[STAGE2_LABEL2ID["process"]]),
            "p_shortcut": float(probs[STAGE2_LABEL2ID["shortcut"]]),
        }


class Stage3Predictor:
    def __init__(self, model_dir: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(DEVICE)
        self.model.eval()
        self.max_length = max_length

        # 按你当前 stage_3 的标签定义固定：
        # Incorrect = 0, Correct = 1
        self.incorrect_idx = 0
        self.correct_idx = 1

    @torch.no_grad()
    def predict_one(self, text: str) -> Dict[str, Any]:
        batch = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = self.model(**batch)
        logits = outputs.logits[0].detach().cpu().numpy()
        probs = softmax_np(np.expand_dims(logits, axis=0))[0]

        pred_id = int(np.argmax(probs))
        pred_label = "Correct" if pred_id == self.correct_idx else "Incorrect"

        return {
            "pred_label": pred_label,
            "logit_incorrect": float(logits[self.incorrect_idx]),
            "logit_correct": float(logits[self.correct_idx]),
            "p_incorrect": float(probs[self.incorrect_idx]),
            "p_correct": float(probs[self.correct_idx]),
        }


# =========================================================
# stage2 聚合特征
# =========================================================
def aggregate_stage2_features(step_preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not step_preds:
        return {
            "num_steps": 0,
            "max_baseline_prob": 0.0,
            "max_epistemic_prob": 0.0,
            "max_process_prob": 0.0,
            "max_shortcut_prob": 0.0,
            "mean_baseline_prob": 0.0,
            "mean_epistemic_prob": 0.0,
            "mean_process_prob": 0.0,
            "mean_shortcut_prob": 0.0,
            "count_pred_baseline": 0,
            "count_pred_epistemic": 0,
            "count_pred_process": 0,
            "count_pred_shortcut": 0,
            "first_epistemic_step_idx": -1,
            "first_process_step_idx": -1,
            "first_shortcut_step_idx": -1,
            "num_non_baseline_steps": 0,
            "major_error_type_id": 0,
            "major_error_confidence": 0.0,
        }

    baseline_probs = [x["p_baseline"] for x in step_preds]
    epistemic_probs = [x["p_epistemic"] for x in step_preds]
    process_probs = [x["p_process"] for x in step_preds]
    shortcut_probs = [x["p_shortcut"] for x in step_preds]

    pred_labels = [x["pred_label"] for x in step_preds]

    count_pred_baseline = sum(1 for x in pred_labels if x == "baseline")
    count_pred_epistemic = sum(1 for x in pred_labels if x == "epistemic")
    count_pred_process = sum(1 for x in pred_labels if x == "process")
    count_pred_shortcut = sum(1 for x in pred_labels if x == "shortcut")

    def find_first_step(target: str) -> int:
        for i, lab in enumerate(pred_labels, start=1):
            if lab == target:
                return i
        return -1

    max_non_baseline = {
        "epistemic": max(epistemic_probs) if epistemic_probs else 0.0,
        "process": max(process_probs) if process_probs else 0.0,
        "shortcut": max(shortcut_probs) if shortcut_probs else 0.0,
    }
    major_error_type = max(max_non_baseline.items(), key=lambda x: x[1])[0]
    major_error_confidence = float(max_non_baseline[major_error_type])

    if major_error_confidence <= (max(baseline_probs) if baseline_probs else 0.0):
        if count_pred_epistemic + count_pred_process + count_pred_shortcut == 0:
            major_error_type = "baseline"
            major_error_confidence = float(max(baseline_probs) if baseline_probs else 0.0)

    return {
        "num_steps": len(step_preds),

        "max_baseline_prob": float(max(baseline_probs)),
        "max_epistemic_prob": float(max(epistemic_probs)),
        "max_process_prob": float(max(process_probs)),
        "max_shortcut_prob": float(max(shortcut_probs)),

        "mean_baseline_prob": mean(baseline_probs),
        "mean_epistemic_prob": mean(epistemic_probs),
        "mean_process_prob": mean(process_probs),
        "mean_shortcut_prob": mean(shortcut_probs),

        "count_pred_baseline": count_pred_baseline,
        "count_pred_epistemic": count_pred_epistemic,
        "count_pred_process": count_pred_process,
        "count_pred_shortcut": count_pred_shortcut,

        "first_epistemic_step_idx": find_first_step("epistemic"),
        "first_process_step_idx": find_first_step("process"),
        "first_shortcut_step_idx": find_first_step("shortcut"),

        "num_non_baseline_steps": (
            count_pred_epistemic + count_pred_process + count_pred_shortcut
        ),
        "major_error_type_id": MAJOR_ERROR_TYPE_ID[major_error_type],
        "major_error_confidence": major_error_confidence,
    }


# =========================================================
# 构造一条融合样本
# =========================================================
def build_one_fusion_row(
    row: Dict[str, Any],
    stage2_predictor: Stage2Predictor,
    stage3_predictor: Stage3Predictor,
) -> Dict[str, Any]:
    sample_id = safe_text(row.get("sample_id"))
    example_id = safe_text(row.get("example_id"))
    condition = safe_text(row.get("condition"))

    premise = safe_text(row.get("premise"))
    question = safe_text(row.get("question"))
    conclusion = safe_text(row.get("conclusion"))
    reasoning_steps = get_reasoning_steps(row)

    raw_label = safe_text(row.get("label"))
    if raw_label in FUSION_LABEL_MAP:
        fusion_label = FUSION_LABEL_MAP[raw_label]
    else:
        fusion_label = CONDITION_TO_BINARY.get(condition, "Incorrect")

    # stage_2：逐步预测
    step_preds: List[Dict[str, Any]] = []
    previous_steps: List[str] = []

    for step_idx, current_step in enumerate(reasoning_steps, start=1):
        stage2_text = build_stage2_input(
            premise=premise,
            question=question,
            previous_steps=previous_steps,
            current_step=current_step,
        )
        pred = stage2_predictor.predict_one(stage2_text)
        pred["step_idx"] = step_idx
        step_preds.append(pred)
        previous_steps.append(current_step)

    stage2_feats = aggregate_stage2_features(step_preds)

    # stage_3：整条预测
    stage3_text = build_stage3_input(
        premise=premise,
        question=question,
        reasoning_steps=reasoning_steps,
        conclusion=conclusion,
    )
    stage3_pred = stage3_predictor.predict_one(stage3_text)

    step_lens = [safe_len_text(s) for s in reasoning_steps]

    out = {
        "sample_id": sample_id,
        "example_id": example_id,
        "condition": condition,
        "fusion_label": fusion_label,

        **stage2_feats,

        "p_correct": float(stage3_pred["p_correct"]),
        "p_incorrect": float(stage3_pred["p_incorrect"]),
        "logit_correct": float(stage3_pred["logit_correct"]),
        "logit_incorrect": float(stage3_pred["logit_incorrect"]),

        "premise_len": safe_len_text(premise),
        "question_len": safe_len_text(question),
        "conclusion_len": safe_len_text(conclusion),
        "avg_step_len": mean(step_lens),
        "max_step_len": max(step_lens) if step_lens else 0,
    }
    return out


def build_dataset(
    base_file: str,
    out_file: str,
    stage2_predictor: Stage2Predictor,
    stage3_predictor: Stage3Predictor,
) -> None:
    print(f"\n========== 开始构造: {out_file} ==========")
    rows = read_jsonl(base_file)
    out_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        if idx % 50 == 0:
            print(f"已处理 {idx}/{len(rows)}")
        out_rows.append(
            build_one_fusion_row(
                row=row,
                stage2_predictor=stage2_predictor,
                stage3_predictor=stage3_predictor,
            )
        )

    write_jsonl(out_rows, out_file)
    print(f"写出完成: {out_file}")
    print(f"样本数: {len(out_rows)}")

    label_count: Dict[str, int] = {}
    for r in out_rows:
        lab = r["fusion_label"]
        label_count[lab] = label_count.get(lab, 0) + 1
    print(f"标签分布: {label_count}")


def main() -> int:
    print(f"DEVICE = {DEVICE}")

    print("加载 stage_2 模型...")
    stage2_predictor = Stage2Predictor(
        STAGE2_MODEL_DIR,
        max_length=STAGE2_MAX_LENGTH,
    )

    print("加载 stage_3 模型...")
    stage3_predictor = Stage3Predictor(
        STAGE3_MODEL_DIR,
        max_length=STAGE3_MAX_LENGTH,
    )

    build_dataset(
        base_file=BASE_TRAIN_FILE,
        out_file=OUT_TRAIN_FILE,
        stage2_predictor=stage2_predictor,
        stage3_predictor=stage3_predictor,
    )

    build_dataset(
        base_file=BASE_DEV_FILE,
        out_file=OUT_DEV_FILE,
        stage2_predictor=stage2_predictor,
        stage3_predictor=stage3_predictor,
    )

    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())