# Part 10 — Benchmarking & Cost Analysis

This report consolidates runtime, memory, throughput, and cost benchmarking for the Project LANTERN pipelines (open-source vs. DocAI).

## 📑 Filing-Level Runtime & Memory (Dual Pipeline)

### NVIDIA_10-K_2024-02-21
- **Pages**: 85
- **Open-source tables**: 72
- **DocAI tables**: 78
- **Runtime**: 317.39 sec
- **Memory**: 286.75 MB

### NVIDIA_10-K_2025-02-26
- **Pages**: 87
- **Open-source tables**: 73
- **DocAI tables**: 78
- **Runtime**: 293.14 sec
- **Memory**: 302.83 MB

## 📊 Stage-wise Performance & Global Metrics

# 📊 Benchmarks & Cost Analysis Report

## Global Summary
- **Total pages processed**: 85
- **Total runtime**: 29.08 seconds
- **Average runtime per page**: 0.342 seconds
- **Average memory usage per page**: 4.54 MB

## Stage-wise Performance
| Stage | Avg Time (s) | Avg Mem (MB) | Success Rate | Samples |
|-------|--------------|--------------|--------------|---------|
| images | 0.0001 | 0.00 | 2.4% | 85 |
| layout | 0.0057 | -0.05 | 100.0% | 85 |
| tables_camelot | 0.1675 | 5.73 | 100.0% | 85 |
| tables_pdfplumber | 0.1688 | -1.15 | 100.0% | 85 |
| text | 0.0000 | 0.00 | 0.0% | 85 |

## Cloud OCR Cost Estimates (100k pages)
- **AWS Textract (OCR only)**: ~$150
- **Google Document AI**: ~$1,500
- **Azure Form Recognizer**: ~$100

## Bottleneck Analysis
- **OCR**: Not included in this run (no stage logged as OCR), but in prior runs OCR dominates time (2–3s/page).
- **Tables (Camelot/pdfplumber)**: ~0.17s/page, moderate memory (~5MB). Main structured data bottleneck.
- **Layout**: Fast (<0.01s/page), negligible memory.
- **Images**: Extremely cheap (~0s/page), low success (only few pages had extractable images).
- **Text (embedded)**: Essentially 0s/page, but 0% success (all pages needed OCR).

## Hardware & Concurrency Recommendations
- **CPU-bound OCR (Tesseract)**: Multi-core CPUs scale well; allocate ~400MB RAM per worker. For 100k pages/month, 32–64GB RAM and 16–32 cores recommended.
- **GPU-accelerated OCR (RapidOCR, PaddleOCR)**: 5–10× faster than CPU OCR; recommended if GPUs available. One modern GPU can replace ~10–20 CPU workers.
- **Concurrency**: Use 10–20 worker processes with batch sizes of ~10 pages for optimal throughput.
- **Distributed processing**: For millions of pages, distribute across multiple machines or cloud nodes with shared storage.

## Scaling Projections
- At current averages (~0.34s/page), 100k pages would take ~9.5 hours on a single core without OCR.
- With OCR (~2.5s/page), 100k pages would take ~70 hours on a single core.
- Parallelism reduces wall-clock time linearly with cores (up to ~20 workers before I/O contention).

## 🔍 Observations

- DocAI detected slightly more tables than the open-source pipeline (78 vs ~72–73), showing stronger structured table extraction.
- Runtime: open-source pipeline completed each filing in ~5 minutes, while DocAI incurred higher per-batch latency.
- Memory usage was moderate (~280–300 MB per filing).
- Tables remain the bottleneck stage (~0.17 sec/page). OCR not triggered in these runs, but when used it dominates runtime (~2–3 sec/page).

## 💰 Cost & Scaling

- **Cloud OCR (100k pages):** AWS Textract ~$150, GCP DocAI ~$1,500, Azure ~$100.
- **Throughput:** at ~0.34s/page, 100k pages take ~9.5h on single core without OCR; with OCR ~70h. Parallelism reduces wall-clock linearly up to ~20 workers.
- **Hardware:** CPU OCR requires ~400MB RAM/worker; GPU OCR gives 5–10× speedup.
