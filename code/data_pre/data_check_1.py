from __future__ import annotations

import json
import math
from collections import Counter
from statistics import mean, median
from typing import Dict, List, Optional

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON 解析失败: line={line_no}, error={e}")
    return rows


def percentile(sorted_values: List[int], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    k = (len(sorted_values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)

    if f == c:
        return float(sorted_values[int(k)])

    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return float(d0 + d1)


def build_bucket_label(x: int, bucket_size: int) -> str:
    low = (x // bucket_size) * bucket_size
    high = low + bucket_size - 1
    return f"{low:>4d}-{high:<4d}"


def summarize_lengths(values: List[int], name: str) -> Dict:
    if not values:
        return {
            "name": name,
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
        }

    values_sorted = sorted(values)
    return {
        "name": name,
        "count": len(values_sorted),
        "min": values_sorted[0],
        "max": values_sorted[-1],
        "mean": round(mean(values_sorted), 2),
        "median": round(median(values_sorted), 2),
        "p90": round(percentile(values_sorted, 0.90), 2),
        "p95": round(percentile(values_sorted, 0.95), 2),
        "p99": round(percentile(values_sorted, 0.99), 2),
    }


def print_summary(summary: Dict) -> None:
    print(f"\n[{summary['name']}]")
    print(f"count   : {summary['count']}")
    print(f"min     : {summary['min']}")
    print(f"max     : {summary['max']}")
    print(f"mean    : {summary['mean']}")
    print(f"median  : {summary['median']}")
    print(f"p90     : {summary['p90']}")
    print(f"p95     : {summary['p95']}")
    print(f"p99     : {summary['p99']}")


def print_histogram(values: List[int], bucket_size: int, title: str) -> None:
    print(f"\n[{title}]")
    if not values:
        print("No data.")
        return

    counter = Counter(build_bucket_label(v, bucket_size) for v in values)
    total = len(values)

    def bucket_key(label: str) -> int:
        return int(label.split("-")[0])

    for bucket in sorted(counter.keys(), key=bucket_key):
        cnt = counter[bucket]
        ratio = cnt / total * 100
        bar = "#" * max(1, int(ratio / 2)) if cnt > 0 else ""
        print(f"{bucket} : {cnt:>6d} ({ratio:>6.2f}%) {bar}")


def top_longest_samples(rows: List[Dict], token_lengths: Optional[List[int]], top_k: int) -> None:
    print(f"\n[Top {top_k} longest premises]")
    if not rows:
        print("No data.")
        return

    items = []
    for i, row in enumerate(rows):
        premise = str(row.get("premise", "") or "")
        item = {
            "idx": i,
            "example_id": row.get("example_id", ""),
            "sample_id": row.get("sample_id", ""),
            "condition": row.get("condition", ""),
            "char_len": len(premise),
            "word_len": len(premise.split()),
            "token_len": token_lengths[i] if token_lengths is not None else None,
            "preview": premise[:120].replace("\n", " "),
        }
        items.append(item)

    if token_lengths is not None:
        items.sort(key=lambda x: (x["token_len"], x["char_len"]), reverse=True)
    else:
        items.sort(key=lambda x: x["char_len"], reverse=True)

    for rank, item in enumerate(items[:top_k], start=1):
        print(
            f"{rank:>2d}. sample_id={item['sample_id']} "
            f"example_id={item['example_id']} "
            f"condition={item['condition']} "
            f"char_len={item['char_len']} "
            f"word_len={item['word_len']} "
            f"token_len={item['token_len']}"
        )
        print(f"    preview: {item['preview']}")


def analyze_file(
    input_path: str,
    tokenizer_name_or_path: Optional[str],
    bucket_size: int,
    top_k: int,
) -> None:
    rows = read_jsonl(input_path)
    print(f"\n==============================")
    print(f"Input file: {input_path}")
    print(f"Total rows : {len(rows)}")

    missing_premise = 0
    empty_premise = 0

    char_lengths = []
    word_lengths = []
    token_lengths = None

    premises = []
    for row in rows:
        premise = row.get("premise", None)
        if premise is None:
            missing_premise += 1
            premise = ""
        premise = str(premise)

        if premise.strip() == "":
            empty_premise += 1

        premises.append(premise)
        char_lengths.append(len(premise))
        word_lengths.append(len(premise.split()))

    print(f"Missing premise field : {missing_premise}")
    print(f"Empty premise content : {empty_premise}")

    print_summary(summarize_lengths(char_lengths, "Premise Character Length"))
    print_summary(summarize_lengths(word_lengths, "Premise Word Length"))

    if tokenizer_name_or_path:
        if AutoTokenizer is None:
            print("\n[WARN] transformers 未安装，无法统计 token 长度。")
        else:
            print(f"\nLoading tokenizer from: {tokenizer_name_or_path}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            token_lengths = []

            for premise in premises:
                enc = tokenizer(
                    premise,
                    truncation=False,
                    add_special_tokens=True,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                token_lengths.append(len(enc["input_ids"]))

            print_summary(summarize_lengths(token_lengths, "Premise Token Length"))
            print_histogram(token_lengths, bucket_size=bucket_size, title="Token Length Histogram")

            # 统计超长比例
            for threshold in [128, 256, 384, 512, 768, 1024]:
                cnt = sum(1 for x in token_lengths if x > threshold)
                ratio = cnt / len(token_lengths) * 100 if token_lengths else 0
                print(f"token_len > {threshold:>4d} : {cnt:>6d} ({ratio:>6.2f}%)")

    print_histogram(char_lengths, bucket_size=bucket_size, title="Character Length Histogram")
    top_longest_samples(rows, token_lengths, top_k=top_k)


def main():
    INPUT_FILES = [
        "./stage2_train.jsonl",
        "./stage2_dev.jsonl",
        "./stage3_train.jsonl",
        "./stage3_dev.jsonl",
    ]

    TOKENIZER_PATH = "/home/pjy/models/roberta-base"  # 不想统计 token 长度就改成 None
    BUCKET_SIZE = 100
    TOP_K = 10

    for input_path in INPUT_FILES:
        analyze_file(
            input_path=input_path,
            tokenizer_name_or_path=TOKENIZER_PATH,
            bucket_size=BUCKET_SIZE,
            top_k=TOP_K,
        )


if __name__ == "__main__":
    main()