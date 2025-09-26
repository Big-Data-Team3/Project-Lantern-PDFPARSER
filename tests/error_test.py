# import json
# from pathlib import Path

# def load_metrics():
#     path = Path(__file__).resolve().parent.parent / "tests" / "metrics_custom_only.json"
#     with open(path) as f:
#         return json.load(f)

# def test_custom_text_wer():
#     metrics = load_metrics()
#     assert metrics["text_custom"]["overall"]["wer"] < 0.5, "Custom extractor WER too high"

# def test_custom_text_cer():
#     metrics = load_metrics()
#     assert metrics["text_custom"]["overall"]["cer"] < 0.3, "Custom extractor CER too high"

# def test_table_precision():
#     metrics = load_metrics()
#     assert metrics["tables_custom_selected"]["overall"]["precision"] >= 0.85,  "Table precision too low"

# def test_table_recall():
#     metrics = load_metrics()
#     assert metrics["tables_custom_selected"]["overall"]["recall"] > 0.85, "Table recall too low"
import json
from pathlib import Path

def load_metrics():
    path = Path(__file__).resolve().parent.parent / "tests" / "metrics_custom_only.json"
    with open(path) as f:
        return json.load(f)

def test_custom_text_wer():
    metrics = load_metrics()
    for doc_id, m in metrics.items():
        assert m["text_custom"]["overall"]["wer"] < 0.5, f"{doc_id}: WER too high"

def test_custom_text_cer():
    metrics = load_metrics()
    for doc_id, m in metrics.items():
        assert m["text_custom"]["overall"]["cer"] < 0.3, f"{doc_id}: CER too high"

def test_table_precision():
    metrics = load_metrics()
    for doc_id, m in metrics.items():
        assert m["tables_custom_selected"]["overall"]["precision"] >= 0.5, f"{doc_id}: Table precision too low"

def test_table_recall():
    metrics = load_metrics()
    for doc_id, m in metrics.items():
        assert m["tables_custom_selected"]["overall"]["recall"] > 0.5, f"{doc_id}: Table recall too low"

