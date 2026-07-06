from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


# =========================================================
# rule_features.py
# =========================================================

CLAIM_CUES = {
    "therefore", "thus", "so", "hence", "because", "implies",
    "means", "conclude", "we know", "it follows", "must be"
}

NEGATION_WORDS = {"not", "no", "never", "none", "cannot", "can't", "isn't", "aren't", "wrong"}


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"[。\n.!?;]+", text)
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    return re.findall(r"[a-z0-9_]+", text)


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def max_sentence_support(step_text: str, premise_text: str) -> float:
    """
    当前步骤与 premise 各句子的最大词项重合度，近似支持度。
    """
    step_tokens = tokenize(step_text)
    if not step_tokens:
        return 0.0

    premise_sents = split_sentences(premise_text)
    if not premise_sents:
        return 0.0

    best = 0.0
    for sent in premise_sents:
        score = jaccard(step_tokens, tokenize(sent))
        if score > best:
            best = score
    return best


def contradiction_score(step_text: str, premise_text: str) -> float:
    """
    非严格 NLI，只做轻量规则近似：
    - step 与 premise 高词重叠
    - 且极性（是否含否定）相反
    """
    step_tokens = tokenize(step_text)
    if not step_tokens:
        return 0.0

    step_has_neg = any(w in NEGATION_WORDS for w in step_tokens)

    best = 0.0
    for sent in split_sentences(premise_text):
        sent_tokens = tokenize(sent)
        overlap = jaccard(step_tokens, sent_tokens)
        if overlap < 0.2:
            continue

        sent_has_neg = any(w in NEGATION_WORDS for w in sent_tokens)
        polarity_flip = float(step_has_neg != sent_has_neg)

        score = overlap * polarity_flip
        if score > best:
            best = score

    return best


def novelty_score(step_text: str, premise_text: str, previous_steps: List[str]) -> float:
    """
    新主张注入：
    当前 step 与 premise 和 previous_steps 都不够接近，但像是在提出一个新结论/新断言。
    """
    step_tokens = tokenize(step_text)
    if not step_tokens:
        return 0.0

    support_from_premise = max_sentence_support(step_text, premise_text)

    prev_best = 0.0
    for s in previous_steps:
        prev_best = max(prev_best, jaccard(step_tokens, tokenize(s)))

    claim_like = any(cue in normalize_text(step_text) for cue in CLAIM_CUES)
    lexical_novel = 1.0 - max(support_from_premise, prev_best)

    score = lexical_novel * (1.0 if claim_like else 0.6)
    return max(0.0, min(1.0, score))


def extract_step_rule_features(
    premise: str,
    question: str,
    previous_steps: List[str],
    current_step: str,
    conclusion: str = "",
) -> Dict[str, float]:
    """
    对单一步骤抽取规则特征。
    """
    support = max_sentence_support(current_step, premise)
    contradiction = contradiction_score(current_step, premise)
    novelty = novelty_score(current_step, premise, previous_steps)

    premise_missing = 1.0 - support
    local_contradiction = contradiction
    new_claim_injection = novelty

    return {
        "rule_support_score": float(support),
        "rule_contradiction_score": float(contradiction),
        "rule_novelty_score": float(novelty),

        "rule_premise_missing_score": float(premise_missing),
        "rule_local_contradiction_score": float(local_contradiction),
        "rule_new_claim_score": float(new_claim_injection),
    }


def aggregate_chain_rule_features(
    premise: str,
    question: str,
    reasoning_steps: List[str],
    conclusion: str = "",
) -> Dict[str, float]:
    """
    从整条推理链抽取聚合规则特征。
    """
    step_feats: List[Dict[str, float]] = []

    for i, step in enumerate(reasoning_steps):
        prev = reasoning_steps[:i]
        feat = extract_step_rule_features(
            premise=premise,
            question=question,
            previous_steps=prev,
            current_step=step,
            conclusion=conclusion,
        )
        step_feats.append(feat)

    if not step_feats:
        return {
            "rule_chain_max_support": 0.0,
            "rule_chain_mean_support": 0.0,
            "rule_chain_max_contradiction": 0.0,
            "rule_chain_mean_contradiction": 0.0,
            "rule_chain_max_novelty": 0.0,
            "rule_chain_mean_novelty": 0.0,

            "rule_first_premise_missing_idx": 0.0,
            "rule_first_contradiction_idx": 0.0,
            "rule_first_new_claim_idx": 0.0,

            "rule_num_flagged_steps": 0.0,
            "rule_major_type_id": 0.0,
            "rule_major_type_score": 0.0,
        }

    def col(key: str) -> List[float]:
        return [x[key] for x in step_feats]

    support_list = col("rule_support_score")
    contradiction_list = col("rule_contradiction_score")
    novelty_list = col("rule_novelty_score")

    def first_idx(vals: List[float], threshold: float) -> float:
        for i, v in enumerate(vals, start=1):
            if v >= threshold:
                return float(i)
        return 0.0

    flagged = 0
    for feat in step_feats:
        if (
            feat["rule_premise_missing_score"] >= 0.7
            or feat["rule_local_contradiction_score"] >= 0.5
            or feat["rule_new_claim_score"] >= 0.6
        ):
            flagged += 1

    type_scores = {
        1: max(x["rule_premise_missing_score"] for x in step_feats),      # 前提支持缺失
        2: max(x["rule_local_contradiction_score"] for x in step_feats),  # 局部矛盾
        3: max(x["rule_new_claim_score"] for x in step_feats),            # 新主张注入
    }
    major_type_id, major_type_score = max(type_scores.items(), key=lambda x: x[1])

    return {
        "rule_chain_max_support": float(max(support_list)),
        "rule_chain_mean_support": float(sum(support_list) / len(support_list)),
        "rule_chain_max_contradiction": float(max(contradiction_list)),
        "rule_chain_mean_contradiction": float(sum(contradiction_list) / len(contradiction_list)),
        "rule_chain_max_novelty": float(max(novelty_list)),
        "rule_chain_mean_novelty": float(sum(novelty_list) / len(novelty_list)),

        "rule_first_premise_missing_idx": first_idx(
            [x["rule_premise_missing_score"] for x in step_feats], threshold=0.7
        ),
        "rule_first_contradiction_idx": first_idx(contradiction_list, threshold=0.5),
        "rule_first_new_claim_idx": first_idx(novelty_list, threshold=0.6),

        "rule_num_flagged_steps": float(flagged),
        "rule_major_type_id": float(major_type_id),
        "rule_major_type_score": float(major_type_score),
    }


# =========================================================
# build_fusion_with_rules.py
# =========================================================

STEP_LABELS = ["baseline", "epistemic", "process", "shortcut"]


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


def text_len(x: Any) -> int:
    return len(str(x).strip().split()) if str(x).strip() else 0


def group_stage2_by_sample(stage2_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in stage2_rows:
        sid = str(row.get("sample_id", "")).strip()
        if sid:
            grouped[sid].append(row)

    for sid in grouped:
        grouped[sid] = sorted(
            grouped[sid],
            key=lambda r: int(r.get("step_id", 10**9)) if str(r.get("step_id", "")).strip() else 10**9
        )
    return dict(grouped)


def map_stage3_by_sample(stage3_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for row in stage3_rows:
        sid = str(row.get("sample_id", "")).strip()
        if sid:
            out[sid] = row
    return out


def aggregate_step_model_features(step_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not step_rows:
        return {
            "num_steps": 0.0,

            "max_baseline_prob": 0.0,
            "max_epistemic_prob": 0.0,
            "max_process_prob": 0.0,
            "max_shortcut_prob": 0.0,

            "mean_baseline_prob": 0.0,
            "mean_epistemic_prob": 0.0,
            "mean_process_prob": 0.0,
            "mean_shortcut_prob": 0.0,

            "count_pred_baseline": 0.0,
            "count_pred_epistemic": 0.0,
            "count_pred_process": 0.0,
            "count_pred_shortcut": 0.0,

            "first_epistemic_step_idx": 0.0,
            "first_process_step_idx": 0.0,
            "first_shortcut_step_idx": 0.0,

            "num_non_baseline_steps": 0.0,
            "major_error_type_id": 0.0,
            "major_error_confidence": 0.0,
        }

    def vals(name: str) -> List[float]:
        return [float(r.get(name, 0.0)) for r in step_rows]

    pb = vals("p_baseline")
    pe = vals("p_epistemic")
    pp = vals("p_process")
    ps = vals("p_shortcut")

    pred_labels = [str(r.get("pred_label", "")).strip() for r in step_rows]

    def first_idx_by_pred(label_name: str) -> float:
        for i, lab in enumerate(pred_labels, start=1):
            if lab == label_name:
                return float(i)
        return 0.0

    count_baseline = sum(1 for x in pred_labels if x == "baseline")
    count_epistemic = sum(1 for x in pred_labels if x == "epistemic")
    count_process = sum(1 for x in pred_labels if x == "process")
    count_shortcut = sum(1 for x in pred_labels if x == "shortcut")

    max_error_dict = {
        1: max(pe) if pe else 0.0,
        2: max(pp) if pp else 0.0,
        3: max(ps) if ps else 0.0,
    }
    major_error_type_id, major_error_confidence = max(max_error_dict.items(), key=lambda x: x[1])

    return {
        "num_steps": float(len(step_rows)),

        "max_baseline_prob": float(max(pb)),
        "max_epistemic_prob": float(max(pe)),
        "max_process_prob": float(max(pp)),
        "max_shortcut_prob": float(max(ps)),

        "mean_baseline_prob": float(np.mean(pb)),
        "mean_epistemic_prob": float(np.mean(pe)),
        "mean_process_prob": float(np.mean(pp)),
        "mean_shortcut_prob": float(np.mean(ps)),

        "count_pred_baseline": float(count_baseline),
        "count_pred_epistemic": float(count_epistemic),
        "count_pred_process": float(count_process),
        "count_pred_shortcut": float(count_shortcut),

        "first_epistemic_step_idx": first_idx_by_pred("epistemic"),
        "first_process_step_idx": first_idx_by_pred("process"),
        "first_shortcut_step_idx": first_idx_by_pred("shortcut"),

        "num_non_baseline_steps": float(count_epistemic + count_process + count_shortcut),
        "major_error_type_id": float(major_error_type_id),
        "major_error_confidence": float(major_error_confidence),
    }


def get_global_model_features(stage3_row: Dict[str, Any]) -> Dict[str, float]:
    if not stage3_row:
        return {
            "p_correct": 0.0,
            "p_incorrect": 0.0,
            "logit_correct": 0.0,
            "logit_incorrect": 0.0,
        }

    return {
        "p_correct": float(stage3_row.get("p_correct", 0.0)),
        "p_incorrect": float(stage3_row.get("p_incorrect", 0.0)),
        "logit_correct": float(stage3_row.get("logit_correct", 0.0)),
        "logit_incorrect": float(stage3_row.get("logit_incorrect", 0.0)),
    }


def build_one_fusion_row(
    raw_row: Dict[str, Any],
    step_rows: List[Dict[str, Any]],
    stage3_row: Dict[str, Any],
) -> Dict[str, Any]:
    premise = str(raw_row.get("premise", "")).strip()
    question = str(raw_row.get("question", "")).strip()
    reasoning_steps = normalize_steps(raw_row.get("reasoning_steps", []))
    conclusion = str(raw_row.get("conclusion", "")).strip()

    model_feat = aggregate_step_model_features(step_rows)
    global_feat = get_global_model_features(stage3_row)
    rule_feat = aggregate_chain_rule_features(
        premise=premise,
        question=question,
        reasoning_steps=reasoning_steps,
        conclusion=conclusion,
    )

    avg_step_len = float(np.mean([text_len(s) for s in reasoning_steps])) if reasoning_steps else 0.0
    max_step_len = float(np.max([text_len(s) for s in reasoning_steps])) if reasoning_steps else 0.0

    fusion_label = str(raw_row.get("label", "")).strip()

    out = {
        "sample_id": str(raw_row.get("sample_id", "")).strip(),
        "example_id": str(raw_row.get("example_id", "")).strip(),
        "condition": str(raw_row.get("condition", "")).strip(),
        "fusion_label": fusion_label,

        "premise_len": float(text_len(premise)),
        "question_len": float(text_len(question)),
        "conclusion_len": float(text_len(conclusion)),
        "avg_step_len": float(avg_step_len),
        "max_step_len": float(max_step_len),
    }

    out.update(model_feat)
    out.update(global_feat)
    out.update(rule_feat)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_file", type=str, required=True, help="原始 stage3 数据文件")
    parser.add_argument("--stage2_pred_file", type=str, required=True, help="stage2 predictions")
    parser.add_argument("--stage3_pred_file", type=str, required=True, help="stage3 predictions")
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    raw_rows = read_jsonl(args.raw_file)
    stage2_rows = read_jsonl(args.stage2_pred_file)
    stage3_rows = read_jsonl(args.stage3_pred_file)

    step_map = group_stage2_by_sample(stage2_rows)
    global_map = map_stage3_by_sample(stage3_rows)

    fusion_rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        sid = str(row.get("sample_id", "")).strip()
        fusion_row = build_one_fusion_row(
            raw_row=row,
            step_rows=step_map.get(sid, []),
            stage3_row=global_map.get(sid, {}),
        )
        fusion_rows.append(fusion_row)

    write_jsonl(args.output_file, fusion_rows)
    print(f"saved to: {args.output_file}")
    print(f"num_rows = {len(fusion_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())