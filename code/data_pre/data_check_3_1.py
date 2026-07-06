from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


# =========================================================
# 路径配置：直接改这里
# =========================================================
BASE_TRAIN_FILE = "./stage3_train.jsonl"
BASE_DEV_FILE = "./stage3_dev.jsonl"

# stage2 逐步预测结果
# 约定：每一行对应某个 sample_id 的某一步
STAGE2_TRAIN_PRED_FILE = "./stage2_train_step_preds.jsonl"
STAGE2_DEV_PRED_FILE = "./stage2_dev_step_preds.jsonl"

# stage3 样本级预测结果
# 约定：每一行对应一个 sample_id
STAGE3_TRAIN_PRED_FILE = "./stage3_train_preds.jsonl"
STAGE3_DEV_PRED_FILE = "./stage3_dev_preds.jsonl"

OUT_TRAIN_FILE = "./fusion_train.jsonl"
OUT_DEV_FILE = "./fusion_dev.jsonl"


# =========================================================
# 标签定义
# =========================================================
FUSION_LABEL_MAP = {
    "Correct": "Correct",
    "Incorrect": "Incorrect",
}

# 如果 base 文件里没有 label，只有 condition，可以启用下面这个映射
CONDITION_TO_BINARY = {
    "baseline": "Correct",
    "epistemic": "Incorrect",
    "process": "Incorrect",
    "shortcut": "Incorrect",
}

STAGE2_LABELS = ["baseline", "epistemic", "process", "shortcut"]
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


def safe_len_text(text: Any) -> int:
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    return len(text.strip())


def get_reasoning_steps(row: Dict[str, Any]) -> List[str]:
    steps = row.get("reasoning_steps", [])
    if not isinstance(steps, list):
        return []
    out: List[str] = []
    for x in steps:
        if x is None:
            continue
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
        else:
            s = str(x).strip()
            if s:
                out.append(s)
    return out


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def argmax_label(prob_dict: Dict[str, float]) -> str:
    best_label = STAGE2_LABELS[0]
    best_value = prob_dict.get(best_label, 0.0)
    for label in STAGE2_LABELS[1:]:
        v = prob_dict.get(label, 0.0)
        if v > best_value:
            best_label = label
            best_value = v
    return best_label


# =========================================================
# 读取 stage2 逐步预测
# 约定输入格式（每一行）例如：
# {
#   "sample_id": "...",
#   "step_idx": 1,
#   "p_baseline": 0.90,
#   "p_epistemic": 0.03,
#   "p_process": 0.04,
#   "p_shortcut": 0.03
# }
# =========================================================
def load_stage2_preds(path: str) -> Dict[str, List[Dict[str, Any]]]:
    rows = read_jsonl(path)
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for r in rows:
        sample_id = str(r.get("sample_id", "")).strip()
        if not sample_id:
            continue

        step_idx = int(r.get("step_idx", 0))
        pred = {
            "step_idx": step_idx,
            "baseline": float(r.get("p_baseline", 0.0)),
            "epistemic": float(r.get("p_epistemic", 0.0)),
            "process": float(r.get("p_process", 0.0)),
            "shortcut": float(r.get("p_shortcut", 0.0)),
        }
        grouped.setdefault(sample_id, []).append(pred)

    for sample_id in grouped:
        grouped[sample_id].sort(key=lambda x: x["step_idx"])

    return grouped


# =========================================================
# 读取 stage3 样本级预测
# 约定输入格式（每一行）例如：
# {
#   "sample_id": "...",
#   "p_correct": 0.12,
#   "p_incorrect": 0.88
# }
# =========================================================
def load_stage3_preds(path: str) -> Dict[str, Dict[str, float]]:
    rows = read_jsonl(path)
    out: Dict[str, Dict[str, float]] = {}

    for r in rows:
        sample_id = str(r.get("sample_id", "")).strip()
        if not sample_id:
            continue

        out[sample_id] = {
            "p_correct": float(r.get("p_correct", 0.0)),
            "p_incorrect": float(r.get("p_incorrect", 0.0)),
        }

    return out


# =========================================================
# 从 stage2 逐步预测聚合出样本级特征
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

    baseline_probs = [x["baseline"] for x in step_preds]
    epistemic_probs = [x["epistemic"] for x in step_preds]
    process_probs = [x["process"] for x in step_preds]
    shortcut_probs = [x["shortcut"] for x in step_preds]

    pred_labels: List[str] = []
    for x in step_preds:
        pred_label = argmax_label({
            "baseline": x["baseline"],
            "epistemic": x["epistemic"],
            "process": x["process"],
            "shortcut": x["shortcut"],
        })
        pred_labels.append(pred_label)

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
# 构造一条 fusion 样本
# base row 约定格式例如：
# {
#   "sample_id": "...",
#   "example_id": "...",
#   "condition": "baseline",
#   "premise": "...",
#   "question": "...",
#   "reasoning_steps": ["...", "..."],
#   "conclusion": "...",
#   "label": "Correct"
# }
# =========================================================
def build_one_fusion_row(
    base_row: Dict[str, Any],
    stage2_grouped: Dict[str, List[Dict[str, Any]]],
    stage3_pred_map: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    sample_id = str(base_row.get("sample_id", "")).strip()
    example_id = str(base_row.get("example_id", "")).strip()
    condition = str(base_row.get("condition", "")).strip()

    premise = str(base_row.get("premise", "") or "").strip()
    question = str(base_row.get("question", "") or "").strip()
    conclusion = str(base_row.get("conclusion", "") or "").strip()
    reasoning_steps = get_reasoning_steps(base_row)

    # 融合标签：优先用 label；没有的话回退到 condition 映射
    raw_label = str(base_row.get("label", "")).strip()
    if raw_label in FUSION_LABEL_MAP:
        fusion_label = FUSION_LABEL_MAP[raw_label]
    else:
        fusion_label = CONDITION_TO_BINARY.get(condition, "Incorrect")

    step_preds = stage2_grouped.get(sample_id, [])
    stage2_feats = aggregate_stage2_features(step_preds)

    stage3_feats = stage3_pred_map.get(
        sample_id,
        {"p_correct": 0.0, "p_incorrect": 0.0},
    )

    step_lens = [safe_len_text(x) for x in reasoning_steps]

    out = {
        "sample_id": sample_id,
        "example_id": example_id,
        "condition": condition,
        "fusion_label": fusion_label,

        # -------- stage2 聚合特征 --------
        **stage2_feats,

        # -------- stage3 特征 --------
        "p_correct": float(stage3_feats.get("p_correct", 0.0)),
        "p_incorrect": float(stage3_feats.get("p_incorrect", 0.0)),

        # -------- 基础统计特征 --------
        "premise_len": safe_len_text(premise),
        "question_len": safe_len_text(question),
        "conclusion_len": safe_len_text(conclusion),
        "avg_step_len": mean(step_lens),
        "max_step_len": max(step_lens) if step_lens else 0,
    }
    return out


def build_dataset(
    base_file: str,
    stage2_pred_file: str,
    stage3_pred_file: str,
    out_file: str,
) -> None:
    print(f"\n========== 开始构造: {out_file} ==========")
    print(f"[1/4] 读取 base 文件: {base_file}")
    base_rows = read_jsonl(base_file)

    print(f"[2/4] 读取 stage2 预测: {stage2_pred_file}")
    stage2_grouped = load_stage2_preds(stage2_pred_file)

    print(f"[3/4] 读取 stage3 预测: {stage3_pred_file}")
    stage3_pred_map = load_stage3_preds(stage3_pred_file)

    print(f"[4/4] 生成 fusion 数据")
    out_rows: List[Dict[str, Any]] = []

    miss_stage2 = 0
    miss_stage3 = 0

    for row in base_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id not in stage2_grouped:
            miss_stage2 += 1
        if sample_id not in stage3_pred_map:
            miss_stage3 += 1

        out_rows.append(build_one_fusion_row(row, stage2_grouped, stage3_pred_map))

    write_jsonl(out_rows, out_file)

    print(f"写出完成: {out_file}")
    print(f"总样本数: {len(out_rows)}")
    print(f"缺少 stage2 预测的样本数: {miss_stage2}")
    print(f"缺少 stage3 预测的样本数: {miss_stage3}")

    label_count: Dict[str, int] = {}
    for r in out_rows:
        lab = r["fusion_label"]
        label_count[lab] = label_count.get(lab, 0) + 1
    print(f"标签分布: {label_count}")


def main() -> int:
    build_dataset(
        base_file=BASE_TRAIN_FILE,
        stage2_pred_file=STAGE2_TRAIN_PRED_FILE,
        stage3_pred_file=STAGE3_TRAIN_PRED_FILE,
        out_file=OUT_TRAIN_FILE,
    )

    build_dataset(
        base_file=BASE_DEV_FILE,
        stage2_pred_file=STAGE2_DEV_PRED_FILE,
        stage3_pred_file=STAGE3_DEV_PRED_FILE,
        out_file=OUT_DEV_FILE,
    )

    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())