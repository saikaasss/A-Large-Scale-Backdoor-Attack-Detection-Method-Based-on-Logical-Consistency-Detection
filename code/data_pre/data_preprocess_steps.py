from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

OUTPUT_DIR = Path("./data")
DEV_RATIO = 0.2
RANDOM_SEED = 42

INPUT_FILES = [
    "baseline.jsonl",
    "shortcut.jsonl",
    "process.jsonl",
    "epistemic.jsonl",
]

# =========================
# 标签定义
# =========================
STAGE2_LABELS = {
    "baseline": "baseline",
    "shortcut": "shortcut",
    "process": "process",
    "epistemic": "epistemic",
}

GLOBAL_LABEL_BY_CONDITION = {
    "baseline": "Correct",
    "shortcut": "Incorrect",
    "process": "Incorrect",
    "epistemic": "Incorrect",
}


# =========================
# 基础 I/O
# =========================
def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# 字段提取
# =========================
def extract_document_and_question(user_prompt: str) -> Tuple[str, str]:
    """
    从 user_prompt 中提取:
    [Document] ... [/Document]
    [Question] ... [/Question]
    """
    if not user_prompt:
        return "", ""

    doc_match = re.search(r"\[Document\]\s*(.*?)\s*\[/Document\]", user_prompt, flags=re.S)
    q_match = re.search(r"\[Question\]\s*(.*?)\s*\[/Question\]", user_prompt, flags=re.S)

    document = doc_match.group(1).strip() if doc_match else ""
    question = q_match.group(1).strip() if q_match else ""

    return document, question


def normalize_reasoning_steps(llm_output_json: Optional[Dict]) -> List[str]:
    """
    llm_output_json 形如:
    {
        "reasoning": [
            {"step": 1, "content": "..."},
            {"step": 2, "content": "..."}
        ],
        "answer": "..."
    }
    """
    if not llm_output_json:
        return []

    reasoning = llm_output_json.get("reasoning", [])
    steps = []

    for item in reasoning:
        if isinstance(item, dict):
            content = str(item.get("content", "")).strip()
            if content:
                steps.append(content)
        elif isinstance(item, str):
            text = item.strip()
            if text:
                steps.append(text)

    return steps


def extract_answer(llm_output_json: Optional[Dict]) -> str:
    if not llm_output_json:
        return ""
    return str(llm_output_json.get("answer", "")).strip()


def is_valid_row(row: Dict) -> bool:
    """
    只保留真正可用的样本:
    - skipped == 0
    - parse_ok == 1
    - llm_output_json 非空
    """
    if row.get("skipped", 1) != 0:
        return False
    if row.get("parse_ok", 0) != 1:
        return False
    if not row.get("llm_output_json"):
        return False
    return True


BASELINE_USE_ANSWER_PROB = 0.3


# =========================
# stage2 当前步骤选择逻辑（四分类）
# =========================
def select_stage2_sample(
    condition: str,
    reasoning_steps: List[str],
    answer: str,
    rng: random.Random,
) -> Optional[Dict]:
    """
    根据 condition 生成一条 stage2 四分类样本。

    规则：
    - baseline:
        随机选取一个 reasoning step 作为 current_step，label = baseline
    - process:
        优先选第 2 步作为 current_step，label = process
    - epistemic:
        优先选第 2 步作为 current_step，label = epistemic
    - shortcut:
        优先选 reasoning 的最后一步作为 current_step，label = shortcut；
        如果 reasoning 为空，则退化为 answer

    返回字段：
    {
        "step_id": int,
        "previous_steps": List[str],
        "current_step": str,
        "label": str,
    }
    """
    if condition == "baseline":
        can_use_reasoning = len(reasoning_steps) > 0
        can_use_answer = bool(answer)

        if not can_use_reasoning and not can_use_answer:
            return None

        # 若 reasoning 和 answer 都有，则按概率决定是否取 answer
        if can_use_reasoning and can_use_answer and rng.random() < BASELINE_USE_ANSWER_PROB:
            return {
                "step_id": len(reasoning_steps) + 1,
                "previous_steps": reasoning_steps,
                "current_step": answer,
                "label": "baseline",
            }

        # 优先走 reasoning step
        if can_use_reasoning:
            idx = rng.randrange(len(reasoning_steps))
            return {
                "step_id": idx + 1,
                "previous_steps": reasoning_steps[:idx],
                "current_step": reasoning_steps[idx],
                "label": "baseline",
            }

        # 没有 reasoning，只有 answer
        return {
            "step_id": 1,
            "previous_steps": [],
            "current_step": answer,
            "label": "baseline",
        }

    if condition == "process":
        if not reasoning_steps:
            return None
        idx = 1 if len(reasoning_steps) >= 2 else 0
        return {
            "step_id": idx + 1,
            "previous_steps": reasoning_steps[:idx],
            "current_step": reasoning_steps[idx],
            "label": "process",
        }

    if condition == "epistemic":
        if not reasoning_steps:
            return None
        idx = 1 if len(reasoning_steps) >= 2 else 0
        return {
            "step_id": idx + 1,
            "previous_steps": reasoning_steps[:idx],
            "current_step": reasoning_steps[idx],
            "label": "epistemic",
        }

    if condition == "shortcut":
        if reasoning_steps:
            idx = len(reasoning_steps) - 1
            return {
                "step_id": idx + 1,
                "previous_steps": reasoning_steps[:idx],
                "current_step": reasoning_steps[idx],
                "label": "shortcut",
            }
        if answer:
            return {
                "step_id": -1,
                "previous_steps": [],
                "current_step": answer,
                "label": "shortcut",
            }
        return None

    return None


# =========================
# 主处理逻辑
# =========================
def convert_raw_rows_to_stage_samples(raw_rows: List[Dict]) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    返回:
    - stage2_rows: 步骤级样本（四分类）
    - stage3_rows: 全局级样本
    - stats: 统计信息
    """
    stage2_rows = []
    stage3_rows = []
    rng = random.Random(RANDOM_SEED)

    stats = {
        "raw_total": 0,
        "valid_rows": 0,
        "dropped_invalid_row": 0,
        "dropped_empty_prompt_fields": 0,
        "dropped_stage2_unselectable": 0,
        "condition_counts": {},
        "stage2_label_counts": {},
        "stage3_label_counts": {},
    }

    for row in raw_rows:
        stats["raw_total"] += 1

        if not is_valid_row(row):
            stats["dropped_invalid_row"] += 1
            continue

        stats["valid_rows"] += 1

        condition = str(row.get("condition", "")).strip()
        example_id = str(row.get("example_id", "")).strip()
        user_prompt = str(row.get("user_prompt", "")).strip()
        llm_output_json = row.get("llm_output_json")

        premise, question = extract_document_and_question(user_prompt)
        reasoning_steps = normalize_reasoning_steps(llm_output_json)
        conclusion = extract_answer(llm_output_json)

        if not premise or not question:
            stats["dropped_empty_prompt_fields"] += 1
            continue

        selected = select_stage2_sample(
            condition=condition,
            reasoning_steps=reasoning_steps,
            answer=conclusion,
            rng=rng,
        )
        if selected is None:
            stats["dropped_stage2_unselectable"] += 1
            continue

        global_label = GLOBAL_LABEL_BY_CONDITION.get(condition, "Correct")
        sample_id = f"{example_id}_{condition}"

        stage2_rows.append(
            {
                "sample_id": sample_id,
                "example_id": example_id,
                "condition": condition,
                "step_id": selected["step_id"],
                "premise": premise,
                "question": question,
                "previous_steps": selected["previous_steps"],
                "current_step": selected["current_step"],
                "label": selected["label"],
            }
        )

        stage3_rows.append(
            {
                "sample_id": sample_id,
                "example_id": example_id,
                "condition": condition,
                "premise": premise,
                "question": question,
                "reasoning_steps": reasoning_steps,
                "conclusion": conclusion,
                "label": global_label,
            }
        )

        stats["condition_counts"][condition] = stats["condition_counts"].get(condition, 0) + 1
        stats["stage2_label_counts"][selected["label"]] = stats["stage2_label_counts"].get(selected["label"], 0) + 1
        stats["stage3_label_counts"][global_label] = stats["stage3_label_counts"].get(global_label, 0) + 1

    return stage2_rows, stage3_rows, stats


def split_ids(sample_ids: List[str], dev_ratio: float, seed: int) -> Tuple[set, set]:
    rng = random.Random(seed)
    sample_ids = sorted(set(sample_ids))
    rng.shuffle(sample_ids)

    if len(sample_ids) == 0:
        return set(), set()

    dev_size = max(1, int(len(sample_ids) * dev_ratio))
    dev_ids = set(sample_ids[:dev_size])
    train_ids = set(sample_ids[dev_size:])
    return train_ids, dev_ids


def split_rows_by_ids(rows: List[Dict], train_ids: set, dev_ids: set) -> Tuple[List[Dict], List[Dict]]:
    train_rows = [row for row in rows if row["sample_id"] in train_ids]
    dev_rows = [row for row in rows if row["sample_id"] in dev_ids]
    return train_rows, dev_rows


def main():
    all_raw_rows = []
    for file_path in INPUT_FILES:
        rows = read_jsonl(file_path)
        all_raw_rows.extend(rows)

    stage2_rows, stage3_rows, stats = convert_raw_rows_to_stage_samples(all_raw_rows)

    shared_sample_ids = [row["sample_id"] for row in stage2_rows]
    train_ids, dev_ids = split_ids(shared_sample_ids, DEV_RATIO, RANDOM_SEED)

    stage2_train, stage2_dev = split_rows_by_ids(stage2_rows, train_ids, dev_ids)
    stage3_train, stage3_dev = split_rows_by_ids(stage3_rows, train_ids, dev_ids)

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(stage2_train)
    rng.shuffle(stage2_dev)
    rng.shuffle(stage3_train)
    rng.shuffle(stage3_dev)

    write_jsonl(OUTPUT_DIR / "stage2_train.jsonl", stage2_train)
    write_jsonl(OUTPUT_DIR / "stage2_dev.jsonl", stage2_dev)
    write_jsonl(OUTPUT_DIR / "stage3_train.jsonl", stage3_train)
    write_jsonl(OUTPUT_DIR / "stage3_dev.jsonl", stage3_dev)

    print("Done.")
    print(f"raw_total:               {stats['raw_total']}")
    print(f"valid_rows:              {stats['valid_rows']}")
    print(f"dropped_invalid_row:     {stats['dropped_invalid_row']}")
    print(f"dropped_empty_prompt:    {stats['dropped_empty_prompt_fields']}")
    print(f"dropped_unselectable:    {stats['dropped_stage2_unselectable']}")
    print(f"stage2_train:            {len(stage2_train)}")
    print(f"stage2_dev:              {len(stage2_dev)}")
    print(f"stage3_train:            {len(stage3_train)}")
    print(f"stage3_dev:              {len(stage3_dev)}")
    print(f"condition_counts:        {stats['condition_counts']}")
    print(f"stage2_label_counts:     {stats['stage2_label_counts']}")
    print(f"stage3_label_counts:     {stats['stage3_label_counts']}")
    print(f"output dir:              {OUTPUT_DIR}")


if __name__ == "__main__":
    main()