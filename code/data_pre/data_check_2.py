from __future__ import annotations

import json
from pathlib import Path
from typing import List

from transformers import AutoTokenizer


# ========= 预设参数，直接改这里 =========
INPUT_FILES = [
    "./stage2_train.jsonl",
    "./stage2_dev.jsonl",
    "./stage3_train.jsonl",
    "./stage3_dev.jsonl",
]

TOKENIZER_PATH = "/home/pjy/models/roberta-base"

MIN_TOKENS = 100
MAX_TOKENS = 400

OUTPUT_SUFFIX = f".premise_tok_{MIN_TOKENS}_{MAX_TOKENS}.jsonl"
# ======================================


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON 解析失败，文件={path} 行号={line_no}: {e}")
    return rows


def save_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_premise_token_len(tokenizer, premise: str) -> int:
    if not premise:
        return 0
    # 不加 special tokens，纯统计 premise 自身长度
    return len(tokenizer.encode(premise, add_special_tokens=False))


def filter_by_premise_length(
    rows: List[dict],
    tokenizer,
    min_tokens: int,
    max_tokens: int,
) -> List[dict]:
    kept = []
    missing_premise = 0
    empty_premise = 0

    for row in rows:
        premise = row.get("premise", None)

        if premise is None:
            missing_premise += 1
            continue

        premise = str(premise).strip()
        if not premise:
            empty_premise += 1
            continue

        token_len = get_premise_token_len(tokenizer, premise)

        if min_tokens <= token_len <= max_tokens:
            kept.append(row)

    print(f"Missing premise : {missing_premise}")
    print(f"Empty premise   : {empty_premise}")
    print(f"Kept rows       : {len(kept)}")

    return kept


def main() -> None:
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    for input_path_str in INPUT_FILES:
        input_path = Path(input_path_str)
        if not input_path.exists():
            print(f"\n[SKIP] 文件不存在: {input_path}")
            continue

        print("\n" + "=" * 60)
        print(f"Input file: {input_path}")

        rows = load_jsonl(input_path)
        print(f"Total rows      : {len(rows)}")

        kept_rows = filter_by_premise_length(
            rows=rows,
            tokenizer=tokenizer,
            min_tokens=MIN_TOKENS,
            max_tokens=MAX_TOKENS,
        )

        output_path = input_path.with_name(input_path.stem + OUTPUT_SUFFIX)
        save_jsonl(output_path, kept_rows)

        keep_ratio = (len(kept_rows) / len(rows) * 100) if rows else 0.0
        print(f"Keep ratio      : {keep_ratio:.2f}%")
        print(f"Output file     : {output_path}")


if __name__ == "__main__":
    main()