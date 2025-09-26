#!/usr/bin/env python
"""
Part 11 - XBRL Validation for NVIDIA Filings
--------------------------------------------
- Parse XBRL instance document (NVIDIA 10-K)
- Load PDF-extracted tables (CSV or Excel from pipeline)
- Cross-verify numeric values
- Handle scaling/sign issues
- Generate validation reports (CSV + Markdown)

Usage:
    python xbrl_validate.py [doc_id]
    If doc_id is omitted, the most recent folder in ../data/parsed is used.
"""

import re
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
import sys

# === GLOBAL PATHS ===
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PARSED_DIR = BASE_DIR / "data" / "parsed"
XBRL_VALIDATION_DIR = BASE_DIR / "data" / "xbrl_validation"


# Ensure required dirs exist
PARSED_DIR.mkdir(parents=True, exist_ok=True)
XBRL_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def parse_xbrl_instance(xbrl_file):
    """Parse XBRL instance file and extract numeric facts"""
    tree = ET.parse(xbrl_file)
    root = tree.getroot()

    facts = []
    for elem in root.iter():
        if elem.text and elem.text.strip():
            txt = elem.text.replace(",", "").strip()
            if re.match(r"^-?\d+(\.\d+)?$", txt):
                try:
                    facts.append({
                        "concept": elem.tag.split("}")[-1],
                        "value": float(txt)
                    })
                except:
                    continue
    return pd.DataFrame(facts)


def load_pdf_tables(doc_id, parsed_dir=PARSED_DIR):
    """Load extracted tables (CSV or Excel) for a given doc_id"""
    tables_dir = parsed_dir / doc_id / "tables"
    dfs = []

    if not tables_dir.exists():
        print(f"✗ No tables directory found for {doc_id}")
        return dfs

    for file in tables_dir.glob("*"):
        try:
            if file.suffix.lower() == ".csv":
                df = pd.read_csv(file)
            elif file.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(file)
            else:
                continue
            if not df.empty:
                dfs.append((file.stem, df))
        except Exception as e:
            print(f"  ✗ Failed to load {file.name}: {e}")
            continue
    return dfs


def extract_number(cell):
    """Extract numeric value from a PDF table cell"""
    if pd.isna(cell):
        return None
    val_str = str(cell).strip().replace(",", "")
    if re.match(r"^\(?-?\d+(\.\d+)?\)?$", val_str):
        try:
            value = float(val_str.strip("()"))
            if val_str.startswith("(") and val_str.endswith(")"):
                value = -value
            return value
        except:
            return None
    return None


def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100


def cross_verify(xbrl_df, pdf_tables):
    """Cross-verify PDF values against XBRL facts"""
    results = []

    for concept, xbrl_val in zip(xbrl_df["concept"], xbrl_df["value"]):
        for table_name, df in pdf_tables:
            for _, row in df.iterrows():
                label = str(row.iloc[0])
                for cell in row[1:]:
                    pdf_val = extract_number(cell)
                    if pdf_val is None:
                        continue

                    # Try scaling factors
                    matched = False
                    for factor in [1, 1000, 1_000_000]:
                        if abs(pdf_val * factor - xbrl_val) < max(1e-2, abs(xbrl_val) * 0.001):
                            results.append({
                                "concept": concept,
                                "xbrl_value": xbrl_val,
                                "pdf_value": pdf_val,
                                "label": label,
                                "table": table_name,
                                "scaling": factor,
                                "similarity": similarity(label, concept),
                                "match_quality": "EXACT" if abs(pdf_val * factor - xbrl_val) < 1e-6 else "CLOSE"
                            })
                            matched = True
                            break
                    if not matched:
                        # Record as mismatch if very close label but wrong value
                        if similarity(label, concept) > 50:
                            results.append({
                                "concept": concept,
                                "xbrl_value": xbrl_val,
                                "pdf_value": pdf_val,
                                "label": label,
                                "table": table_name,
                                "scaling": None,
                                "similarity": similarity(label, concept),
                                "match_quality": "MISMATCH"
                            })
    return pd.DataFrame(results)


def generate_report(doc_id, results_df):
    """Generate CSV + Markdown validation report"""
    out_dir = XBRL_VALIDATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = out_dir / f"{doc_id}_xbrl_validation.csv"
    results_df.to_csv(csv_path, index=False)

    # Markdown summary
    exact = len(results_df[results_df["match_quality"] == "EXACT"])
    close = len(results_df[results_df["match_quality"] == "CLOSE"])
    mismatch = len(results_df[results_df["match_quality"] == "MISMATCH"])

    lines = [
        f"# XBRL Validation Report for {doc_id}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        f"- Concepts validated: {len(results_df)}",
        f"- Exact matches: {exact}",
        f"- Close matches: {close}",
        f"- Mismatches: {mismatch}",
        "",
        "## Sample Results",
    ]

    for _, row in results_df.head(20).iterrows():
        lines.append(
            f"- **{row['concept']}** → PDF: {row['pdf_value']} | "
            f"XBRL: {row['xbrl_value']} | Quality: {row['match_quality']}"
        )

    md_path = out_dir / f"{doc_id}_xbrl_validation.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved results: {csv_path}")
    print(f"Saved report: {md_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    # Allow doc_id to be passed or auto-detect the most recent
    if len(sys.argv) >= 2:
        doc_id = sys.argv[1]
    else:
        # Exclude validation/output folders from candidate doc_ids
        candidates = [
            d for d in PARSED_DIR.iterdir()
            if d.is_dir() and d.name != "xbrl_validation"
        ]

        if not candidates:
            print(f"No parsed documents found in {PARSED_DIR}")
            sys.exit(1)
        # Pick most recent folder
        doc_id = max(candidates, key=lambda d: d.stat().st_mtime).name
        print(f"No doc_id provided, using most recent: {doc_id}")

    # Prefer instance documents (usually *_htm.xml)
    xbrl_files = list(RAW_DIR.rglob(f"{doc_id}/*_htm.xml"))

    if not xbrl_files:  # fallback: any XML
        xbrl_files = list(RAW_DIR.rglob(f"{doc_id}/*.xml"))

        if not xbrl_files:
            print(f"No XBRL file found for {doc_id}")
            sys.exit(1)

    xbrl_file = xbrl_files[0]


    print(f"\nParsing XBRL: {xbrl_file}")
    xbrl_df = parse_xbrl_instance(xbrl_file)
    print(f"Parsed {len(xbrl_df)} facts from XBRL")

    print("\nLoading PDF tables...")
    pdf_tables = load_pdf_tables(doc_id)
    print(f"Loaded {len(pdf_tables)} tables")

    print("\nCross-verifying PDF vs XBRL...")
    results_df = cross_verify(xbrl_df, pdf_tables)
    print(f"Found {len(results_df)} potential matches")

    print("\nGenerating reports...")
    generate_report(doc_id, results_df)

    print("\nValidation complete")


if __name__ == "__main__":
    main()
