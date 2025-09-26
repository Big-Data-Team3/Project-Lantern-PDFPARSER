# Project Lantern — SEC Filings PDF Parser

## Overview

Project Lantern is a **modular, reproducible pipeline** for parsing SEC 10-K filings into structured outputs.  
It integrates:  
- **Custom parser** (pdfplumber, Camelot, PyMuPDF, OCR with Tesseract)  
- **Docling with RapidOCR fallback**  
- **Google DocAI (triggered within text_extractor_all)**  

The outputs are benchmarked against **ground-truth datasets** and validated with **XBRL filings**. Tests ensure regression checks on accuracy.  

---

## Project Resources

- **Pipeline (DVC)**: run with `dvc repro`  
- **Ground-truth**: `data/ground_truth` for evaluation  
- **Evaluation**: `src/error_rate_detection.py` generates metrics  
- **Tests**: `pytest` with `tests/error_test.py`  
- **Validation**: `src/xbrl_validate.py` cross-verifies against SEC XBRL  

---

## Environment Setup

### 1. Clone & Install
```bash
git clone <repo-url>
cd PROJECT-LANTERN
pip install -r requirements.txt
```

### 2. Environment Variables (`.env`)
Create a `.env` file at the root with:

```env
# Required for SEC filings download & PDF conversion
SEC_API_KEY=your-sec-api-token

# Used by sec_data_extraction.py for HTTP headers
COMPANY_NAME=YourNameOrOrg
EMAIL=your-email@example.com

# Google Cloud credentials for Document AI
GOOGLE_APPLICATION_CREDENTIALS=secrets/gcloud-key.json
```

### 3. Google Cloud Key
- Download your **service account JSON** for Document AI.  
- Place it in:  
  ```
  secrets/gcloud-key.json
  ```  
- Make sure `.gitignore` excludes it (never commit keys to GitHub).  

---

## Technologies

![Python](https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![pdfplumber](https://img.shields.io/badge/-pdfplumber-000000?style=for-the-badge)
![Camelot](https://img.shields.io/badge/-Camelot-FF6600?style=for-the-badge)
![PyMuPDF](https://img.shields.io/badge/-PyMuPDF-1E90FF?style=for-the-badge)
![Tesseract OCR](https://img.shields.io/badge/-Tesseract%20OCR-5C2D91?style=for-the-badge)
![Docling](https://img.shields.io/badge/-Docling-00A36C?style=for-the-badge)
![Google Cloud](https://img.shields.io/badge/-Google%20Cloud-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)
![DVC](https://img.shields.io/badge/-DVC-945DD6?style=for-the-badge&logo=dvc)
![Pytest](https://img.shields.io/badge/-Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas)

---

## Project Flow

### Step 0: Ingestion (`sec_data_extraction.py`)
- Download SEC 10-K filings (NVIDIA default).  
- Convert `.htm` filings to PDFs using **SEC API** (`SEC_API_KEY`).  
- Download **XBRL attachments**.  

### Step 1–3: Custom Parser (`text_extractor_all.py`)
- **Text extraction** → pdfplumber + OCR fallback (`ocr_config.py`).  
- **Table extraction** → Camelot + pdfplumber fallback.  
- **Layout extraction** → PyMuPDF with bounding boxes and fonts.  
- **DocAI integration** → triggered inside parser for comparison.  
- Outputs: `.txt`, `.csv`, `.json`, `.md`.  

### Step 4: Docling (`docling_rapidocr_final.py`)
- Run Docling + RapidOCR fallback.  
- Exports JSON, Markdown, text, tables, and images.  

### Step 5–6: Metadata & Storage
- Provenance in `.jsonl` and Markdown.  
- Exports combined `.json`, `.md`, `.txt`.  

### Step 7: Managed Service (inside `text_extractor_all.py`)
- DocAI triggered within parsing stage for baseline outputs.  

### Step 8: Orchestration
- DVC pipeline orchestrates all stages.  

### Step 9: Testing (`error_rate_detection.py` + `error_test.py`)
- **WER/CER** metrics for text.  
- **Precision/recall** metrics for tables.  
- Thresholds enforced by pytest (`error_test.py`).  

### Step 10: Benchmarking
- Runtime + memory logging per stage.  
- Summary in `data/parsed/summary_dual.json`.  

### Step 11: XBRL Validation (`xbrl_validate.py`)
- Parse XBRL facts.  
- Match against extracted tables with scaling/sign handling.  
- Reports saved in `data/xbrl_validation/`.  

---

## ⚙️ DVC Pipeline

Our project is orchestrated with **Data Version Control (DVC)** to ensure reproducibility of all stages.  
The pipeline runs end-to-end from downloading raw SEC filings to validation against XBRL.

### Pipeline Stages
```yaml
stages:
  download:
    cmd: python src/sec_data_extraction.py
    deps:
      - src/sec_data_extraction.py
      - src/config.py
    outs:
      - data/raw

  parse:
    cmd: python src/text_extractor_all.py
    deps:
      - src/text_extractor_all.py
      - src/ocr_config.py
      - src/docai_integration.py
      - data/raw
    outs:
      - data/parsed

  evaluate:
    cmd: python src/error_rate_detection.py
    deps:
      - src/error_rate_detection.py
      - data/parsed
      - data/ground_truth
    outs:
      - tests/metrics_custom_only.json

  xbrl_validate:
    cmd: python src/xbrl_validate.py
    deps:
      - src/xbrl_validate.py
      - data/raw
      - data/parsed
    outs:
      - data/xbrl_validation
```

---

## Repository Structure

```bash
PROJECT-LANTERN-PDFPARSER/
├── data/
│   ├── raw/              # SEC filings (PDF + XBRL)
│   ├── parsed/           # pipeline outputs
│   ├── ground_truth/     # truth sets for evaluation
├── src/
│   ├── config.py
│   ├── ocr_config.py
│   ├── sec_data_extraction.py
│   ├── text_extractor_all.py
│   ├── docling_rapidocr_final.py
│   ├── docai_integration.py
│   ├── error_rate_detection.py
│   ├── xbrl_validate.py
│   └── utils/
├── tests/
│   ├── error_test.py
│   ├── metrics_custom_only.json
│   └── .gitkeep
├── secrets/
│   └── gcloud-key.json (NOT committed, add to .gitignore)
├── reports/
│   └── benchmark_report.md
├── dvc.yaml
├── requirements.txt
├── README.md
└── .env.example
```

---

## 📘 Codelab
Our full step-by-step walkthrough is published here:  
👉 [Open Codelab](https://codelabs-preview.appspot.com/?file_id=1Ks-2sDrYw8s72LZVT1aIB7zW4f0vxWpnEIMCNa2xSCE#0)

## 🎥 Demo Video

Watch the full walkthrough of **Project LANTERN** here:  

👉 [YouTube Demo](https://youtu.be/emJRSmvVHEw)

[![Project LANTERN Demo](https://img.youtube.com/vi/emJRSmvVHEw/0.jpg)](https://youtu.be/emJRSmvVHEw)


## Contributions

| Member                        | Contribution                                                                 |
| ----------------------------- | --------------------------------------------------------------------------- |
| **Krittivasan V Jagannathan** | 33% - Custom parser (text, OCR, tables, layout — Parts 1–3, 5–6), benchmarking     |
| **Sthavir Gokul Soroff**      | 33% - SEC ingestion + XBRL download (Part 0, 11), validation & testing             |
| **Pooja Subramanya Raghu**    | 33% - Docling + DocAI integration (Parts 4, 7), DVC orchestration, evaluation flow |

---

## Attestation

WE ATTEST THAT WE HAVEN’T USED ANY OTHER STUDENTS’ WORK IN OUR  
ASSIGNMENT AND ABIDE BY THE POLICIES LISTED IN THE STUDENT HANDBOOK.
