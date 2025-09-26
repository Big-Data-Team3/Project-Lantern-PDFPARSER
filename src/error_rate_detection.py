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
# PARSED_DIR = BASE_PATH / "parsed"


# # === HELPERS ===
# def safe_read(path):
#     return path.read_text(encoding="utf-8", errors="ignore").strip()


# # === TEXT METRIC FUNCTION ===
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


# # === TABLE METRIC FUNCTION ===
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

#             # Align rows and columns
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
#     aggregated_all = {}

#     # Iterate over all parsed documents
#     for doc_dir in PARSED_DIR.iterdir():
#         if not doc_dir.is_dir():
#             continue

#         doc_id = doc_dir.name
#         print(f"\n=== Evaluating {doc_id} ===")

#         text_dir = doc_dir / "text"
#         table_dir = doc_dir / "tables"

#         if not text_dir.exists() and not table_dir.exists():
#             print(f"No text/tables found in {doc_id}, skipping.")
#             continue

#         text_scores = evaluate_text_metrics(text_dir) if text_dir.exists() else {}
#         table_scores = evaluate_selected_table_metrics(GT_TABLE_DIR, table_dir) if table_dir.exists() else {}

#         aggregated_all[doc_id] = {
#             "text_custom": {
#                 "overall": {
#                     "wer": np.mean([v["wer"] for v in text_scores.values()]) if text_scores else float("nan"),
#                     "cer": np.mean([v["cer"] for v in text_scores.values()]) if text_scores else float("nan")
#                 },
#                 "per_page": text_scores
#             },
#             "tables_custom_selected": {
#                 "overall": {
#                     "precision": np.mean([v["precision"] for v in table_scores.values()]) if table_scores else float("nan"),
#                     "recall": np.mean([v["recall"] for v in table_scores.values()]) if table_scores else float("nan")
#                 },
#                 "per_page": table_scores
#             }
#         }

#     # Save aggregated results
#     output_path = Path(__file__).resolve().parent.parent / "tests" / "metrics_custom_only.json"
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(aggregated_all, f, indent=2)

#     print("\nFinal Evaluation Metrics (Custom Parser Only, All Docs):")
#     print(json.dumps(aggregated_all, indent=2))


# if __name__ == "__main__":
#     main()

# # from pathlib import Path
# # import json
# # import numpy as np
# # import pandas as pd
# # from jiwer import wer, cer
# # import csv

# # # === CONFIG ===
# # TEXT_PAGES = [f"{i:03d}" for i in range(1, 11)]
# # TABLE_PAGES = ["012", "032", "040", "050", "051", "052"]

# # # BASE PATH (relative to project root)
# # BASE_PATH = Path(__file__).resolve().parent.parent / "data"

# # # === PATH DEFINITIONS ===
# # GT_TEXT_DIR = BASE_PATH / "ground_truth" / "text"
# # GT_TABLE_DIR = BASE_PATH / "ground_truth" / "tables"
# # PARSED_DIR = BASE_PATH / "parsed"


# # # === HELPERS ===
# # def safe_read(path):
# #     """Read text safely across platforms (handles BOM, encoding quirks)."""
# #     try:
# #         return path.read_text(encoding="utf-8-sig", errors="ignore").strip()
# #     except Exception as e:
# #         print(f"⚠️ Failed to read {path}: {e}")
# #         return ""


# # def safe_read_csv(path):
# #     """
# #     Try to read a CSV robustly across OS and malformed rows.
# #     Returns a pandas DataFrame (may be empty if all parsing fails).
# #     """
# #     try:
# #         # First attempt: strict parsing
# #         return pd.read_csv(
# #             path,
# #             header=None,
# #             encoding="utf-8-sig",
# #             dtype=str,
# #             sep=",",
# #             engine="python",
# #             quoting=3
# #         ).fillna("")
# #     except Exception as e1:
# #         print(f"⚠️ Strict parse failed for {path}: {e1}")
# #         try:
# #             # Second attempt: Python's CSV reader with sniffer
# #             with open(path, newline="", encoding="utf-8-sig") as f:
# #                 sample = f.read(2048)
# #                 f.seek(0)
# #                 try:
# #                     dialect = csv.Sniffer().sniff(sample)
# #                 except Exception:
# #                     dialect = csv.excel
# #                 reader = csv.reader(f, dialect)
# #                 rows = [row for row in reader]
# #             return pd.DataFrame(rows).fillna("")
# #         except Exception as e2:
# #             print(f"⚠️ Sniffer parse failed for {path}: {e2}")
# #             try:
# #                 # Third attempt: raw line split, pad uneven rows
# #                 rows = []
# #                 max_cols = 0
# #                 with open(path, encoding="utf-8-sig", errors="ignore") as f:
# #                     for line in f:
# #                         parts = [p.strip() for p in line.strip().split(",")]
# #                         max_cols = max(max_cols, len(parts))
# #                         rows.append(parts)
# #                 # Pad rows
# #                 norm_rows = [r + [""] * (max_cols - len(r)) for r in rows]
# #                 return pd.DataFrame(norm_rows).fillna("")
# #             except Exception as e3:
# #                 print(f"❌ Could not parse {path}: {e3}")
# #                 return pd.DataFrame()


# # # === TEXT METRIC FUNCTION ===
# # def evaluate_text_metrics(source_dir, gt_dir):
# #     """Compare ground truth vs predicted text files."""
# #     metrics = {}
# #     for page_num in TEXT_PAGES:
# #         gt_candidates = [
# #             gt_dir / f"page_{page_num}.txt",
# #             gt_dir / f"page{page_num}.txt"
# #         ]
# #         pred_candidates = [
# #             source_dir / f"page_{page_num}.txt",
# #             source_dir / f"page{page_num}.txt"
# #         ]

# #         gt_file = next((f for f in gt_candidates if f.exists()), None)
# #         pred_file = next((f for f in pred_candidates if f.exists()), None)

# #         print(f"Checking GT: {gt_file} | Prediction: {pred_file}")
# #         if gt_file and pred_file:
# #             gt_text = safe_read(gt_file)
# #             pred_text = safe_read(pred_file)

# #             if gt_text and pred_text:
# #                 metrics[f"page_{page_num}"] = {
# #                     "wer": wer(gt_text, pred_text),
# #                     "cer": cer(gt_text, pred_text)
# #                 }
# #             else:
# #                 print(f"⚠️ Empty text in page_{page_num}, skipping.")
# #         else:
# #             print(f"Missing file(s) for page_{page_num}. Skipping.")
# #     return metrics


# # # === TABLE METRIC FUNCTION ===
# # def evaluate_selected_table_metrics(gt_dir, pred_dir):
# #     """Compare ground truth vs predicted tables."""
# #     metrics = {}
# #     for page_num in TABLE_PAGES:
# #         gt_candidates = [
# #             gt_dir / f"page_{page_num}.csv",
# #             gt_dir / f"page{page_num}.csv"
# #         ]
# #         pred_candidates = [
# #             pred_dir / f"page_{page_num}_best.csv",
# #             pred_dir / f"page{page_num}_best.csv"
# #         ]

# #         gt_file = next((f for f in gt_candidates if f.exists()), None)
# #         pred_file = next((f for f in pred_candidates if f.exists()), None)

# #         print(f"Checking TABLE GT: {gt_file} | Prediction: {pred_file}")
# #         if gt_file and pred_file:
# #             gt_df = safe_read_csv(gt_file)
# #             pred_df = safe_read_csv(pred_file)

# #             if gt_df.empty or pred_df.empty:
# #                 print(f"⚠️ Skipping page {page_num}, could not parse tables.")
# #                 continue

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


# # # === MAIN ===
# # def main():
# #     aggregated_all = {}

# #     # Iterate over all parsed documents
# #     for doc_dir in PARSED_DIR.iterdir():
# #         if not doc_dir.is_dir():
# #             continue

# #         doc_id = doc_dir.name
# #         print(f"\n=== Evaluating {doc_id} ===")

# #         text_dir = doc_dir / "text"
# #         table_dir = doc_dir / "tables"

# #         if not text_dir.exists() and not table_dir.exists():
# #             print(f"No text/tables found in {doc_id}, skipping.")
# #             continue

# #         text_scores = evaluate_text_metrics(text_dir, GT_TEXT_DIR) if text_dir.exists() else {}
# #         table_scores = evaluate_selected_table_metrics(GT_TABLE_DIR, table_dir) if table_dir.exists() else {}

# #         aggregated_all[doc_id] = {
# #             "text_custom": {
# #                 "overall": {
# #                     "wer": np.mean([v["wer"] for v in text_scores.values()]) if text_scores else float("nan"),
# #                     "cer": np.mean([v["cer"] for v in text_scores.values()]) if text_scores else float("nan")
# #                 },
# #                 "per_page": text_scores
# #             },
# #             "tables_custom_selected": {
# #                 "overall": {
# #                     "precision": np.mean([v["precision"] for v in table_scores.values()]) if table_scores else float("nan"),
# #                     "recall": np.mean([v["recall"] for v in table_scores.values()]) if table_scores else float("nan")
# #                 },
# #                 "per_page": table_scores
# #             }
# #         }

# #     # Save aggregated results (ensure folder exists)
# #     output_path = Path(__file__).resolve().parent.parent / "tests" / "metrics_custom_only.json"
# #     output_path.parent.mkdir(parents=True, exist_ok=True)

# #     with open(output_path, "w", encoding="utf-8") as f:
# #         json.dump(aggregated_all, f, indent=2)

# #     print("\nFinal Evaluation Metrics (Custom Parser Only, All Docs):")
# #     print(json.dumps(aggregated_all, indent=2))


# # if __name__ == "__main__":
# #     main()

from pathlib import Path
import json
import numpy as np
import pandas as pd
from jiwer import wer, cer
import platform

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
    """Safe text read with BOM handling"""
    return path.read_text(encoding="utf-8-sig", errors="ignore").strip()


def robust_read_csv(path):
    system = platform.system()
    try:
        if system == "Windows":
            return pd.read_csv(
                path,
                header=None,
                encoding="utf-8-sig",
                dtype=str
            ).fillna("")
        else:
            return pd.read_csv(
                path,
                header=None,
                encoding="utf-8-sig",
                dtype=str,
                sep=",",
                engine="python",
                quoting=3
            ).fillna("")
    except Exception as e:
        print(f"pandas.read_csv failed for {path.name}, trying fallback. Reason: {e}")
        # === Fallback: Manual CSV parsing ===
        rows = []
        with open(path, encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                row = [cell.strip() for cell in line.strip().split(",")]
                rows.append(row)
        return pd.DataFrame(rows)

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
            try:
                gt_df = robust_read_csv(gt_file)
                pred_df = robust_read_csv(pred_file)
            except Exception as e:
                print(f"Failed to parse CSVs for page {page_num}: {e}")
                continue

            if gt_df.empty or pred_df.empty:
                print(f"Empty DataFrame for page {page_num}, skipping.")
                continue

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

    # Save aggregated results (ensure tests dir exists)
    output_path = Path(__file__).resolve().parent.parent / "tests" / "metrics_custom_only.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_all, f, indent=2)

    print("\nFinal Evaluation Metrics (Custom Parser Only, All Docs):")
    print(json.dumps(aggregated_all, indent=2))


if __name__ == "__main__":
    main()

