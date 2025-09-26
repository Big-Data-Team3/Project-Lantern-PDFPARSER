# from pathlib import Path
# import json
# import numpy as np
# import pandas as pd
# from jiwer import wer, cer

# # === CONFIG ===
# TEXT_PAGES = [f"{i:03d}" for i in range(1, 11)]
# TABLE_PAGES = ["012", "032", "040", "050", "051", "052"]

# # BASE PATH (relative to project root)
# BASE_PATH = Path(__file__).resolve().parent.parent / "data"

# # === PATH DEFINITIONS ===
# GT_TEXT_DIR = BASE_PATH / "ground_truth" / "text"
# GT_TABLE_DIR = BASE_PATH / "ground_truth" / "tables"
# CUSTOM_TEXT_DIR = BASE_PATH / "parsed" / "NVIDIA_10-K_2024-02-21" / "text"
# CUSTOM_TABLE_DIR = BASE_PATH / "parsed" / "NVIDIA_10-K_2024-02-21" / "tables"


# # === TEXT METRIC FUNCTION ===
# # def evaluate_text_metrics(source_dir):
# #     metrics = {}
# #     for page_num in TEXT_PAGES:
# #         gt_file = GT_TEXT_DIR / f"page_{page_num}.txt"
# #         pred_file = source_dir / f"page{page_num}.txt"

# #         print(f"Checking GT: {gt_file} | Prediction: {pred_file}")
# #         if gt_file.exists() and pred_file.exists():
# #             gt_text = gt_file.read_text().strip()
# #             pred_text = pred_file.read_text().strip()
# #             metrics[f"page_{page_num}"] = {
# #                 "wer": wer(gt_text, pred_text),
# #                 "cer": cer(gt_text, pred_text)
# #             }
# #         else:
# #             print(f"Missing file(s) for page_{page_num} in text comparison. Skipping.")
# #     return metrics

# def safe_read(path):
#     return path.read_text(encoding="utf-8", errors="ignore").strip()

# def evaluate_text_metrics(source_dir):
#     metrics = {}
#     for page_num in TEXT_PAGES:
#         gt_candidates = [
#             GT_TEXT_DIR / f"page_{page_num}.txt",
#             GT_TEXT_DIR / f"page{page_num}.txt"
#         ]
#         pred_candidates = [
#             source_dir / f"page_{page_num}.txt",
#             source_dir / f"page{page_num}.txt"
#         ]

#         gt_file = next((f for f in gt_candidates if f.exists()), None)
#         pred_file = next((f for f in pred_candidates if f.exists()), None)

#         print(f"Checking GT: {gt_file} | Prediction: {pred_file}")
#         if gt_file and pred_file:
#             gt_text = safe_read(gt_file)
#             pred_text = safe_read(pred_file)
#             metrics[f"page_{page_num}"] = {
#                 "wer": wer(gt_text, pred_text),
#                 "cer": cer(gt_text, pred_text)
#             }
#         else:
#             print(f"Missing file(s) for page_{page_num}. Skipping.")
#     return metrics


# # === TABLE METRIC FUNCTION (Custom Parser Only, Selected Pages) ===
# # def evaluate_selected_table_metrics(gt_dir, pred_dir):
# #     metrics = {}
# #     for page_num in TABLE_PAGES:
# #         gt_file = gt_dir / f"page_{page_num}.csv"
# #         pred_file = pred_dir / f"page{page_num}_best.csv"

# #         print(f"Checking TABLE GT: {gt_file} | Prediction: {pred_file}")
# #         if gt_file.exists() and pred_file.exists():
# #             gt_df = pd.read_csv(gt_file, header=None).fillna("").astype(str)
# #             pred_df = pd.read_csv(pred_file, header=None).fillna("").astype(str)

# #             # Align rows and columns
# #             gt_df, pred_df = gt_df.align(pred_df, join="outer", axis=0, fill_value="")
# #             gt_df, pred_df = gt_df.align(pred_df, join="outer", axis=1, fill_value="")

# #             print(f"Shape GT: {gt_df.shape}, Pred: {pred_df.shape}")

# #             tp = np.sum(gt_df.values == pred_df.values)
# #             total_pred = np.prod(pred_df.shape)
# #             total_gt = np.prod(gt_df.shape)

# #             metrics[f"page_{page_num}"] = {
# #                 "precision": tp / total_pred if total_pred else 0.0,
# #                 "recall": tp / total_gt if total_gt else 0.0
# #             }
# #         else:
# #             print(f"Missing file(s) for page_{page_num} in table comparison. Skipping.")
# #     return metrics

# def evaluate_selected_table_metrics(gt_dir, pred_dir):
#     metrics = {}
#     for page_num in TABLE_PAGES:
#         gt_candidates = [
#             gt_dir / f"page_{page_num}.csv",
#             gt_dir / f"page{page_num}.csv"
#         ]
#         pred_candidates = [
#             pred_dir / f"page_{page_num}_best.csv",
#             pred_dir / f"page{page_num}_best.csv"
#         ]

#         gt_file = next((f for f in gt_candidates if f.exists()), None)
#         pred_file = next((f for f in pred_candidates if f.exists()), None)

#         print(f"Checking TABLE GT: {gt_file} | Prediction: {pred_file}")
#         if gt_file and pred_file:
#             gt_df = pd.read_csv(gt_file, header=None, encoding="utf-8", dtype=str).fillna("")
#             pred_df = pd.read_csv(pred_file, header=None, encoding="utf-8", dtype=str).fillna("")
#             #Align rows and columns
#             gt_df, pred_df = gt_df.align(pred_df, join="outer", axis=0, fill_value="")
#             gt_df, pred_df = gt_df.align(pred_df, join="outer", axis=1, fill_value="")

#             print(f"Shape GT: {gt_df.shape}, Pred: {pred_df.shape}")

#             tp = np.sum(gt_df.values == pred_df.values)
#             total_pred = np.prod(pred_df.shape)
#             total_gt = np.prod(gt_df.shape)

#             metrics[f"page_{page_num}"] = {
#                 "precision": tp / total_pred if total_pred else 0.0,
#                 "recall": tp / total_gt if total_gt else 0.0
#             }
#         else:
#             print(f"Missing file(s) for page_{page_num} in table comparison. Skipping.")
#     return metrics

# # === MAIN ===
# def main():
#     text_scores = evaluate_text_metrics(CUSTOM_TEXT_DIR)
#     table_scores = evaluate_selected_table_metrics(GT_TABLE_DIR, CUSTOM_TABLE_DIR)

#     aggregated = {
#         "text_custom": {
#             "overall": {
#                 "wer": np.mean([v["wer"] for v in text_scores.values()]) if text_scores else float("nan"),
#                 "cer": np.mean([v["cer"] for v in text_scores.values()]) if text_scores else float("nan")
#             },
#             "per_page": text_scores
#         },
#         "tables_custom_selected": {
#             "overall": {
#                 "precision": np.mean([v["precision"] for v in table_scores.values()]) if table_scores else float("nan"),
#                 "recall": np.mean([v["recall"] for v in table_scores.values()]) if table_scores else float("nan")
#             },
#             "per_page": table_scores
#         }
#     }

#     output_path = Path(__file__).resolve().parent.parent / "tests" / "metrics_custom_only.json"
#     with open(output_path, "w") as f:
#         json.dump(aggregated, f, indent=2)

#     print("\nFinal Evaluation Metrics (Custom Parser Only):")
#     print(json.dumps(aggregated, indent=2))

# if __name__ == "__main__":
#     main()

from pathlib import Path
import json
import numpy as np
import pandas as pd
from jiwer import wer, cer

# === CONFIG ===
TEXT_PAGES = [f"{i:03d}" for i in range(1, 11)]
TABLE_PAGES = ["012", "032", "040", "050", "051", "052"]

# BASE PATH (relative to project root)
BASE_PATH = Path(__file__).resolve().parent.parent / "data"

# === PATH DEFINITIONS ===
GT_TEXT_DIR = BASE_PATH / "ground_truth" / "text"
GT_TABLE_DIR = BASE_PATH / "ground_truth" / "tables"
PARSED_DIR = BASE_PATH / "parsed"


# === HELPERS ===
def safe_read(path):
    return path.read_text(encoding="utf-8", errors="ignore").strip()


# === TEXT METRIC FUNCTION ===
def evaluate_text_metrics(source_dir):
    metrics = {}
    for page_num in TEXT_PAGES:
        gt_candidates = [
            GT_TEXT_DIR / f"page_{page_num}.txt",
            GT_TEXT_DIR / f"page{page_num}.txt"
        ]
        pred_candidates = [
            source_dir / f"page_{page_num}.txt",
            source_dir / f"page{page_num}.txt"
        ]

        gt_file = next((f for f in gt_candidates if f.exists()), None)
        pred_file = next((f for f in pred_candidates if f.exists()), None)

        print(f"Checking GT: {gt_file} | Prediction: {pred_file}")
        if gt_file and pred_file:
            gt_text = safe_read(gt_file)
            pred_text = safe_read(pred_file)
            metrics[f"page_{page_num}"] = {
                "wer": wer(gt_text, pred_text),
                "cer": cer(gt_text, pred_text)
            }
        else:
            print(f"Missing file(s) for page_{page_num}. Skipping.")
    return metrics


# === TABLE METRIC FUNCTION ===
def evaluate_selected_table_metrics(gt_dir, pred_dir):
    metrics = {}
    for page_num in TABLE_PAGES:
        gt_candidates = [
            gt_dir / f"page_{page_num}.csv",
            gt_dir / f"page{page_num}.csv"
        ]
        pred_candidates = [
            pred_dir / f"page_{page_num}_best.csv",
            pred_dir / f"page{page_num}_best.csv"
        ]

        gt_file = next((f for f in gt_candidates if f.exists()), None)
        pred_file = next((f for f in pred_candidates if f.exists()), None)

        print(f"Checking TABLE GT: {gt_file} | Prediction: {pred_file}")
        if gt_file and pred_file:
            gt_df = pd.read_csv(gt_file, header=None, encoding="utf-8", dtype=str).fillna("")
            pred_df = pd.read_csv(pred_file, header=None, encoding="utf-8", dtype=str).fillna("")

            # Align rows and columns
            gt_df, pred_df = gt_df.align(pred_df, join="outer", axis=0, fill_value="")
            gt_df, pred_df = gt_df.align(pred_df, join="outer", axis=1, fill_value="")

            print(f"Shape GT: {gt_df.shape}, Pred: {pred_df.shape}")

            tp = np.sum(gt_df.values == pred_df.values)
            total_pred = np.prod(pred_df.shape)
            total_gt = np.prod(gt_df.shape)

            metrics[f"page_{page_num}"] = {
                "precision": tp / total_pred if total_pred else 0.0,
                "recall": tp / total_gt if total_gt else 0.0
            }
        else:
            print(f"Missing file(s) for page_{page_num} in table comparison. Skipping.")
    return metrics


# === MAIN ===
def main():
    aggregated_all = {}

    # Iterate over all parsed documents
    for doc_dir in PARSED_DIR.iterdir():
        if not doc_dir.is_dir():
            continue

        doc_id = doc_dir.name
        print(f"\n=== Evaluating {doc_id} ===")

        text_dir = doc_dir / "text"
        table_dir = doc_dir / "tables"

        if not text_dir.exists() and not table_dir.exists():
            print(f"No text/tables found in {doc_id}, skipping.")
            continue

        text_scores = evaluate_text_metrics(text_dir) if text_dir.exists() else {}
        table_scores = evaluate_selected_table_metrics(GT_TABLE_DIR, table_dir) if table_dir.exists() else {}

        aggregated_all[doc_id] = {
            "text_custom": {
                "overall": {
                    "wer": np.mean([v["wer"] for v in text_scores.values()]) if text_scores else float("nan"),
                    "cer": np.mean([v["cer"] for v in text_scores.values()]) if text_scores else float("nan")
                },
                "per_page": text_scores
            },
            "tables_custom_selected": {
                "overall": {
                    "precision": np.mean([v["precision"] for v in table_scores.values()]) if table_scores else float("nan"),
                    "recall": np.mean([v["recall"] for v in table_scores.values()]) if table_scores else float("nan")
                },
                "per_page": table_scores
            }
        }

    # Save aggregated results
    output_path = Path(__file__).resolve().parent.parent / "tests" / "metrics_custom_only.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_all, f, indent=2)

    print("\nFinal Evaluation Metrics (Custom Parser Only, All Docs):")
    print(json.dumps(aggregated_all, indent=2))


if __name__ == "__main__":
    main()

