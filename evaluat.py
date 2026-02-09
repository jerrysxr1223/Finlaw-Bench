import argparse
import re

import pandas as pd


SPLIT_PATTERN = re.compile(r"[;,/、；，]+")


def normalize_text(s):
    if pd.isna(s):
        return ""
    return str(s).strip()


def split_multi_label(s):
    s = normalize_text(s)
    if not s:
        return set()
    parts = SPLIT_PATTERN.split(s)
    return {p.strip() for p in parts if p and p.strip()}


def parse_legal_pairs(text):
    text = "" if pd.isna(text) else str(text)
    pairs = set()
    law_names = set()

    law_pattern = re.compile(r"《(.*?)》([^《]*)")
    for m in law_pattern.finditer(text):
        law = m.group(1).strip()
        tail = m.group(2)
        if law:
            law_names.add(law)
        for art in re.findall(r"(第[零一二三四五六七八九十百千\d]+条)", tail):
            art_norm = art.strip()
            if law and art_norm:
                pairs.add((law, art_norm))

    return pairs, law_names


def grade_metrics(y_true, y_pred, labels):
    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in labels and p in labels:
            cm[t][p] += 1

    per_label = {}
    for lbl in labels:
        tp = cm[lbl][lbl]
        fp = sum(cm[t][lbl] for t in labels if t != lbl)
        fn = sum(cm[lbl][p] for p in labels if p != lbl)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_label[lbl] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    macro_f1 = sum(per_label[l]["f1"] for l in labels) / len(labels)

    total = sum(sum(cm[t].values()) for t in labels)
    tp_total = sum(cm[l][l] for l in labels)
    fp_total = sum(sum(cm[t][p] for t in labels) for p in labels) - tp_total
    fn_total = sum(sum(cm[t][p] for p in labels) for t in labels) - tp_total
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)) if (micro_precision + micro_recall) else 0.0
    accuracy = tp_total / total if total else 0.0

    return {
        "confusion_matrix": cm,
        "per_label": per_label,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "accuracy": accuracy,
    }


def evaluate(pred_df, true_df):
    pred_df = pred_df.copy()
    true_df = true_df.copy()

    pred_df["序号"] = pred_df["序号"].astype(str).str.strip()
    true_df["序号"] = true_df["序号"].astype(str).str.strip()

    merged = pred_df.merge(true_df, on="序号", how="left", suffixes=("_pred", "_true"))

    details_rows = []

    type_scores = []
    field_scores = []
    law_scores = []
    grade_corrects = []

    jaccard_type = []
    jaccard_field = []
    hamming_type = []
    hamming_field = []

    type_tp = type_fp = type_fn = 0
    field_tp = field_fp = field_fn = 0
    law_pair_tp = law_pair_fp = law_pair_fn = 0
    law_name_hits = 0

    grade_true = []
    grade_pred = []

    for _, row in merged.iterrows():
        pred_type_set = split_multi_label(row.get("处罚类型_pred", ""))
        true_type_set = split_multi_label(row.get("处罚类型_true", ""))

        pred_field_set = split_multi_label(row.get("违规领域_pred", ""))
        true_field_set = split_multi_label(row.get("违规领域_true", ""))

        pred_pairs, pred_laws = parse_legal_pairs(row.get("法律依据_pred", ""))
        true_pairs, true_laws = parse_legal_pairs(row.get("法律依据_true", ""))

        if pred_type_set == true_type_set:
            type_score = 2
        elif pred_type_set & true_type_set:
            type_score = 1
        else:
            type_score = 0

        if pred_field_set == true_field_set:
            field_score = 2
        elif pred_field_set & true_field_set:
            field_score = 1
        else:
            field_score = 0

        if pred_pairs == true_pairs and len(true_pairs) > 0:
            law_score = 2
        elif pred_laws & true_laws:
            law_score = 1
        else:
            law_score = 0

        pred_grade = normalize_text(row.get("处罚等级", "")).upper()
        true_grade = normalize_text(row.get("罚没金额分组", "")).upper()
        grade_correct = 1 if pred_grade == true_grade else 0

        details_rows.append(
            {
                "序号": row.get("序号", ""),
                "处罚类型分数": type_score,
                "违规领域分数": field_score,
                "法律依据分数": law_score,
                "处罚等级是否正确": grade_correct,
            }
        )

        type_scores.append(type_score)
        field_scores.append(field_score)
        law_scores.append(law_score)
        grade_corrects.append(grade_correct)

        union_type = pred_type_set | true_type_set
        inter_type = pred_type_set & true_type_set
        jaccard_type.append(len(inter_type) / len(union_type) if union_type else 0.0)
        hamming_type.append(len(pred_type_set ^ true_type_set) / len(union_type) if union_type else 0.0)

        union_field = pred_field_set | true_field_set
        inter_field = pred_field_set & true_field_set
        jaccard_field.append(len(inter_field) / len(union_field) if union_field else 0.0)
        hamming_field.append(len(pred_field_set ^ true_field_set) / len(union_field) if union_field else 0.0)

        type_tp += len(pred_type_set & true_type_set)
        type_fp += len(pred_type_set - true_type_set)
        type_fn += len(true_type_set - pred_type_set)

        field_tp += len(pred_field_set & true_field_set)
        field_fp += len(pred_field_set - true_field_set)
        field_fn += len(true_field_set - pred_field_set)

        law_pair_tp += len(pred_pairs & true_pairs)
        law_pair_fp += len(pred_pairs - true_pairs)
        law_pair_fn += len(true_pairs - pred_pairs)
        if pred_laws & true_laws:
            law_name_hits += 1

        if true_grade in {"A", "B", "C", "D", "E"}:
            grade_true.append(true_grade)
            grade_pred.append(pred_grade if pred_grade in {"A", "B", "C", "D", "E"} else "")

    details_df = pd.DataFrame(details_rows)

    def safe_div(a, b):
        return a / b if b else 0.0

    n = len(details_rows)
    type_exact_rate = safe_div(sum(1 for s in type_scores if s == 2), n)
    type_partial_rate = safe_div(sum(1 for s in type_scores if s == 1), n)
    type_wrong_rate = safe_div(sum(1 for s in type_scores if s == 0), n)

    field_exact_rate = safe_div(sum(1 for s in field_scores if s == 2), n)
    field_partial_rate = safe_div(sum(1 for s in field_scores if s == 1), n)
    field_wrong_rate = safe_div(sum(1 for s in field_scores if s == 0), n)

    law_exact_rate = safe_div(sum(1 for s in law_scores if s == 2), n)
    law_partial_rate = safe_div(sum(1 for s in law_scores if s == 1), n)
    law_wrong_rate = safe_div(sum(1 for s in law_scores if s == 0), n)

    type_precision = safe_div(type_tp, (type_tp + type_fp))
    type_recall = safe_div(type_tp, (type_tp + type_fn))
    type_f1 = safe_div(2 * type_precision * type_recall, (type_precision + type_recall))

    field_precision = safe_div(field_tp, (field_tp + field_fp))
    field_recall = safe_div(field_tp, (field_tp + field_fn))
    field_f1 = safe_div(2 * field_precision * field_recall, (field_precision + field_recall))

    law_pair_precision = safe_div(law_pair_tp, (law_pair_tp + law_pair_fp))
    law_pair_recall = safe_div(law_pair_tp, (law_pair_tp + law_pair_fn))
    law_pair_f1 = safe_div(2 * law_pair_precision * law_pair_recall, (law_pair_precision + law_pair_recall))
    law_name_hit_rate = safe_div(law_name_hits, n)

    grade_accuracy = safe_div(sum(grade_corrects), n)

    grade_metrics_result = grade_metrics(grade_true, grade_pred, ["A", "B", "C", "D", "E"]) if grade_true else {}

    metrics_rows = [
        {"metric": "处罚类型_完全正确率", "value": type_exact_rate},
        {"metric": "处罚类型_部分正确率", "value": type_partial_rate},
        {"metric": "处罚类型_错误率", "value": type_wrong_rate},
        {"metric": "处罚类型_Precision", "value": type_precision},
        {"metric": "处罚类型_Recall", "value": type_recall},
        {"metric": "处罚类型_F1", "value": type_f1},
        {"metric": "处罚类型_Jaccard平均", "value": sum(jaccard_type) / n if n else 0.0},
        {"metric": "处罚类型_HammingLoss", "value": sum(hamming_type) / n if n else 0.0},

        {"metric": "违规领域_完全正确率", "value": field_exact_rate},
        {"metric": "违规领域_部分正确率", "value": field_partial_rate},
        {"metric": "违规领域_错误率", "value": field_wrong_rate},
        {"metric": "违规领域_Precision", "value": field_precision},
        {"metric": "违规领域_Recall", "value": field_recall},
        {"metric": "违规领域_F1", "value": field_f1},
        {"metric": "违规领域_Jaccard平均", "value": sum(jaccard_field) / n if n else 0.0},
        {"metric": "违规领域_HammingLoss", "value": sum(hamming_field) / n if n else 0.0},

        {"metric": "法律依据_完全正确率", "value": law_exact_rate},
        {"metric": "法律依据_部分正确率", "value": law_partial_rate},
        {"metric": "法律依据_错误率", "value": law_wrong_rate},
        {"metric": "法律依据_法律名称命中率", "value": law_name_hit_rate},
        {"metric": "法律依据_条款对_Precision", "value": law_pair_precision},
        {"metric": "法律依据_条款对_Recall", "value": law_pair_recall},
        {"metric": "法律依据_条款对_F1", "value": law_pair_f1},

        {"metric": "处罚等级_Accuracy", "value": grade_accuracy},
    ]

    if grade_metrics_result:
        metrics_rows.append({"metric": "处罚等级_MacroF1", "value": grade_metrics_result["macro_f1"]})
        metrics_rows.append({"metric": "处罚等级_MicroF1", "value": grade_metrics_result["micro_f1"]})
        for lbl, vals in grade_metrics_result["per_label"].items():
            metrics_rows.append({"metric": f"处罚等级_{lbl}_Precision", "value": vals["precision"]})
            metrics_rows.append({"metric": f"处罚等级_{lbl}_Recall", "value": vals["recall"]})
            metrics_rows.append({"metric": f"处罚等级_{lbl}_F1", "value": vals["f1"]})

    return details_df, pd.DataFrame(metrics_rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate experiment outputs.")
    parser.add_argument("--pred", default=r"C:\Users\SXR\Desktop\实验及模型评估结果\实验结果\主实验（加入向量数据库）\csv\result_qwen3-max-2026-01-23.csv", help="Prediction CSV path")
    parser.add_argument("--truth", default="data_finlaw.csv", help="Ground truth CSV path")
    parser.add_argument("--out_details", default=r"C:\Users\SXR\Desktop\主实验结果_qwen3-max-2026-01-23_evaluation_details.csv", help="Output details CSV")
    parser.add_argument("--out_metrics", default=r"C:\Users\SXR\Desktop\主实验结果_qwen3-max-2026-01-23_evaluation_metrics.csv", help="Output metrics CSV")
    args = parser.parse_args()

    pred_df = pd.read_csv(args.pred, dtype=str, encoding="utf-8")
    true_df = pd.read_csv(args.truth, dtype=str, encoding="utf-8")

    details_df, metrics_df = evaluate(pred_df, true_df)

    details_df.to_csv(args.out_details, index=False, encoding="utf-8-sig")
    metrics_df.to_csv(args.out_metrics, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
