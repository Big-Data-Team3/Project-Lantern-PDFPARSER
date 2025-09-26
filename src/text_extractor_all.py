# """
# Unified PDF Extractor (Parts 1–6 + Benchmarking + Memory Tracking)
# -------------------------------------------------------------------
# - Part 1: Text extraction with OCR fallback
# - Part 2: Table extraction (Camelot + pdfplumber, scored)
# - Part 3: Layout extraction (PyMuPDF: text, spans, fonts, images, tables)
# - Part 5: Metadata & provenance tagging (JSONL, per-page JSON/MD, global Markdown + JSON with inline tables)
# - Part 6: Storage formats (Markdown, JSON, TXT exports)
# - Benchmarking: runtime + memory consumption per stage (layout, text, OCR, tables, images)
# """

# import json
# import re
# import time
# import pdfplumber
# import pytesseract
# from pdf2image import convert_from_path
# import camelot
# import pandas as pd
# import fitz  # PyMuPDF
# from pathlib import Path
# from ocr_config import test_ocr_setup
# import psutil, os   # <-- NEW for memory tracking

# from pathlib import Path

# PARSED_DIR = Path(__file__).resolve().parents[1] / "data" / "parsed"
# PARSED_DIR.mkdir(parents=True, exist_ok=True)

# def process_with_fallback(pdf_path, output_dir=PARSED_DIR):
#     output_dir = Path(output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)   # <-- ensure parsed/ always exists
#     pdf_path = Path(pdf_path)
#     pdf_name = pdf_path.stem

# # === Optional DocAI import ===
# try:
#     from docai_integration import process_docai
# except ImportError:
#     process_docai = None

# # Toggle for DocAI fallback
# ENABLE_DOCAI = False

# # GCP config (only needed if ENABLE_DOCAI=True)
# PROJECT_ID = "docai-test-473000"
# LOCATION = "us"
# PROCESSOR_ID = "cd1280bfb6a64860"


# # =====================================================
# # Helpers
# # =====================================================
# def int_to_rgb(color_int):
#     if color_int is None:
#         return (0, 0, 0)
#     r = (color_int >> 16) & 255
#     g = (color_int >> 8) & 255
#     b = color_int & 255
#     return (r, g, b)


# def rect_to_list(rect):
#     if rect is None:
#         return []
#     return [rect.x0, rect.y0, rect.x1, rect.y1]


# def decode_font_flags(flags: int):
#     return {
#         "superscript": bool(flags & 1),
#         "italic": bool(flags & 2),
#         "serif": bool(flags & 4),
#         "monospace": bool(flags & 8),
#         "bold": bool(flags & 16),
#     }


# def classify_block(block, page_height):
#     text = " ".join(
#         span["text"] for line in block["lines"] for span in line["spans"]
#     ).strip()
#     if not text:
#         return None, None
#     sizes = [span["size"] for line in block["lines"] for span in line["spans"]]
#     max_size = max(sizes)
#     avg_size = sum(sizes) / len(sizes)
#     x0, y0, x1, y1 = block["bbox"]
#     if y1 < page_height * 0.1:
#         return "header", text
#     if y0 > page_height * 0.9:
#         return "footer", text
#     if max_size >= avg_size * 1.5:
#         return "title", text
#     if max_size >= avg_size * 1.2:
#         return "heading", text
#     return "paragraph", text


# def score_table(df: pd.DataFrame) -> float:
#     if df is None or df.empty:
#         return -1
#     rows, cols = df.shape
#     if rows < 3 or cols < 2:
#         return -1
#     numeric_cells = df.applymap(
#         lambda x: str(x).replace(",", "").replace(".", "").isdigit()
#     ).sum().sum()
#     numeric_ratio = numeric_cells / (rows * cols)
#     return (rows * 0.1) + (cols * 0.2) + (numeric_ratio * 2)


# def parse_filename(doc_id: str):
#     match = re.match(r"([A-Za-z0-9]+).*?(\d{4})", doc_id)
#     if match:
#         company, year = match.groups()
#         return company, int(year)
#     return doc_id, None


# # =====================================================
# # Benchmarking Helpers (time + memory)
# # =====================================================
# process = psutil.Process(os.getpid())

# def get_memory_usage_mb():
#     """Return current memory usage in MB (RSS)."""
#     return process.memory_info().rss / (1024 * 1024)


# def timed_section(stage, func, *args, **kwargs):
#     start_time = time.perf_counter()
#     start_mem = get_memory_usage_mb()
#     try:
#         result = func(*args, **kwargs)
#         ok = True
#     except Exception as e:
#         result, ok = None, False
#     end_time = time.perf_counter()
#     end_mem = get_memory_usage_mb()

#     duration = end_time - start_time
#     mem_used = end_mem - start_mem

#     return result, duration, ok, mem_used


# # =====================================================
# # Unified Runner (Parts 1–6)
# # =====================================================
# def process_pdf(pdf_path, output_dir=PARSED_DIR):
#     pdf_path = Path(pdf_path)
#     pdf_name = pdf_path.stem
#     output_path = Path(output_dir) / pdf_name
#     (output_path / "text").mkdir(parents=True, exist_ok=True)
#     (output_path / "tables").mkdir(parents=True, exist_ok=True)
#     (output_path / "images").mkdir(parents=True, exist_ok=True)
#     (output_path / "json").mkdir(parents=True, exist_ok=True)
#     (output_path / "metadata").mkdir(parents=True, exist_ok=True)

#     # OCR setup
#     ocr_ok, ocr_msg = test_ocr_setup()
#     ocr_available = ocr_ok

#     doc = fitz.open(pdf_path)
#     global_layout = []
#     ocr_pages = []
#     word_boxes = []
#     table_metadata = {}
#     saved_tables = 0

#     # Benchmark logs
#     benchmarks = []

#     # ---- Part 3: Layout detection first ----
#     for page_num, page in enumerate(doc, 1):
#         page_height = page.rect.height
#         page_layout = {
#             "page": page_num,
#             "text_file": f"text/page{page_num:03d}.txt",
#             "blocks": []
#         }

#         # ---- Layout ----
#         text_dict, dur, ok, mem = timed_section("layout", page.get_text, "dict")
#         benchmarks.append({"page": page_num, "stage": "layout", "time_sec": round(dur, 6), "mem_mb": round(mem, 2), "success": ok})
#         if not ok or text_dict is None:
#             text_dict = {"blocks": []}

#         for block in text_dict["blocks"]:
#             if "lines" not in block:
#                 continue
#             block_lines = []
#             for line in block["lines"]:
#                 line_text = "".join(span["text"] for span in line["spans"]).strip()
#                 if not line_text:
#                     continue
#                 block_lines.append({
#                     "text": line_text,
#                     "spans": [
#                         {
#                             "text": span["text"],
#                             "font": span["font"],
#                             "size": span["size"],
#                             "color": int_to_rgb(span.get("color", 0)),
#                             "style": decode_font_flags(span["flags"])
#                         }
#                         for span in line["spans"]
#                     ]
#                 })
#             if block_lines:
#                 block_type, _ = classify_block(block, page_height)
#                 page_layout["blocks"].append({
#                     "type": block_type or "paragraph",
#                     "bbox": list(block["bbox"]),
#                     "lines": block_lines
#                 })

#         # ---- Part 1: Text extraction ----
#         with pdfplumber.open(pdf_path) as pdf:
#             pl_page = pdf.pages[page_num - 1]
#             text, dur, ok, mem = timed_section("text", pl_page.extract_text, 2, 2)
#             benchmarks.append({"page": page_num, "stage": "text", "time_sec": round(dur, 6), "mem_mb": round(mem, 2), "success": bool(text and text.strip())})
#             words = pl_page.extract_words() or []
#             if text and text.strip():
#                 (output_path / f"text/page{page_num:03d}.txt").write_text(text, encoding="utf-8")
#             else:
#                 if ocr_available:
#                     images, dur_img, _, mem_img = timed_section("pdf2image", convert_from_path, pdf_path, first_page=page_num, last_page=page_num)
#                     if images:
#                         ocr_text, dur_ocr, _, mem_ocr = timed_section("ocr", pytesseract.image_to_string, images[0])
#                         (output_path / f"text/page{page_num:03d}.txt").write_text(ocr_text.strip(), encoding="utf-8")
#                         ocr_pages.append(page_num)
#                         benchmarks.append({
#                             "page": page_num,
#                             "stage": "ocr",
#                             "time_sec": round(dur_img + dur_ocr, 6),
#                             "mem_mb": round(mem_img + mem_ocr, 2),
#                             "success": bool(ocr_text.strip())
#                         })
#             if words:
#                 word_boxes.append({"page_num": page_num, "words": words})

#         # ---- Part 2: Table extraction ----
#         candidates = []
#         try:
#             tables, dur, ok, mem = timed_section("camelot", camelot.read_pdf, str(pdf_path), pages=str(page_num), flavor="stream")
#             benchmarks.append({"page": page_num, "stage": "tables_camelot", "time_sec": round(dur, 6), "mem_mb": round(mem, 2), "success": ok and tables is not None and len(tables) > 0})
#             if tables:
#                 for t in tables:
#                     if not t.df.empty:
#                         candidates.append(("camelot_stream", t.df, list(t._bbox)))
#         except Exception:
#             pass
#         try:
#             with pdfplumber.open(pdf_path) as pdf:
#                 pl_page = pdf.pages[page_num - 1]
#                 start = time.perf_counter()
#                 start_mem = get_memory_usage_mb()
#                 for table in pl_page.extract_tables() or []:
#                     if table and len(table) >= 2 and len(table[0]) >= 2:
#                         df = pd.DataFrame(table[1:], columns=table[0])
#                         candidates.append(("pdfplumber", df, list(pl_page.bbox)))
#                 dur = time.perf_counter() - start
#                 mem = get_memory_usage_mb() - start_mem
#                 benchmarks.append({"page": page_num, "stage": "tables_pdfplumber", "time_sec": round(dur, 6), "mem_mb": round(mem, 2), "success": len(candidates) > 0})
#         except Exception:
#             pass

#         scored = []
#         for method, df, bbox in candidates:
#             s = score_table(df)
#             if s > 0:
#                 scored.append((s, method, df, bbox))
#         if scored:
#             best_score, best_method, best_df, best_bbox = max(scored, key=lambda x: x[0])
#             saved_tables += 1
#             filename = f"page{page_num:03d}_best.csv"
#             filepath = output_path / "tables" / filename
#             best_df.to_csv(filepath, index=False, encoding="utf-8-sig")
#             table_metadata[page_num] = {
#                 "csv": f"tables/{filename}",
#                 "bbox": best_bbox,
#                 "rows": best_df.shape[0],
#                 "cols": best_df.shape[1],
#                 "method": best_method,
#                 "score": best_score
#             }
#             page_layout["blocks"].append({"type": "table", **table_metadata[page_num]})

#         # ---- Images ----
#         start = time.perf_counter()
#         start_mem = get_memory_usage_mb()
#         imgs = page.get_images(full=True)
#         dur = time.perf_counter() - start
#         mem = get_memory_usage_mb() - start_mem
#         benchmarks.append({"page": page_num, "stage": "images", "time_sec": round(dur, 6), "mem_mb": round(mem, 2), "success": len(imgs) > 0})
#         for img_idx, img in enumerate(imgs, 1):
#             xref = img[0]
#             bbox = page.get_image_bbox(img)
#             img_path = output_path / "images" / f"page{page_num:03d}_img{img_idx}.png"
#             pix = fitz.Pixmap(doc, xref)
#             if pix.n - pix.alpha < 4:
#                 pix.save(img_path)
#             else:
#                 fitz.Pixmap(fitz.csRGB, pix).save(img_path)
#             page_layout["blocks"].append({"type": "image", "bbox": rect_to_list(bbox), "path": f"images/{img_path.name}"})

#         # Save per-page JSON
#         (output_path / "json" / f"page{page_num:03d}.json").write_text(json.dumps(page_layout, indent=2, ensure_ascii=False), encoding="utf-8")
#         global_layout.append(page_layout)

#     # ---- Save benchmark log ----
#     (output_path / "benchmarks.json").write_text(json.dumps(benchmarks, indent=2), encoding="utf-8")

#     return {
#         "pdf": pdf_name,
#         "pages": len(global_layout),
#         "ocr_pages": len(ocr_pages),
#         "tables": saved_tables,
#         "output_dir": str(output_path)
#     }


# # =====================================================
# # Unified entrypoint with DocAI fallback
# # =====================================================
# def process_with_fallback(pdf_path, output_dir=PARSED_DIR):
#     pdf_path = Path(pdf_path)
#     pdf_name = pdf_path.stem

#     # inside process_with_fallback
#     result = process_pdf(pdf_path, output_dir=output_dir)

#     # Validate extraction
#     text_dir = Path(result["output_dir"]) / "text"
#     has_text = any(text_dir.glob("*.txt"))

#     if has_text:
#         # Always accept open-source result when ENABLE_DOCAI=False
#         result["source"] = "open-source"
#         return result

#     # Only reach here if no text at all
#     if ENABLE_DOCAI and process_docai:
#         print(f"[Pipeline] Falling back to Google DocAI for {pdf_name}...")
#         docai_result = process_docai(pdf_path, output_dir, PROJECT_ID, LOCATION, PROCESSOR_ID)
#         docai_result["pdf"] = pdf_name
#         docai_result["source"] = "docai"
#         return docai_result

#     # print(f"[Pipeline] Skipping DocAI fallback for {pdf_name} (disabled or unavailable).")
#     # return {
#     #     "pdf": pdf_name,
#     #     "pages": 0,
#     #     "ocr_pages": 0,
#     #     "tables": 0,
#     #     "output_dir": str(output_dir / pdf_name),
#     #     "source": "failed"
#     # }
    
#     # If DocAI disabled, just return the open-source result anyway
#     print(f"[Pipeline] Skipping DocAI fallback for {pdf_name} (disabled or unavailable).")
#     result["source"] = "open-source"
#     return result


# # =====================================================
# # Main
# # =====================================================
# if __name__ == "__main__":
#     for pdf_file in (Path(__file__).resolve().parents[1] / "data" / "raw").glob("*.pdf"):
#         print(f"\n=== Processing {pdf_file.name} ===")
#         result = process_with_fallback(pdf_file, output_dir=PARSED_DIR)
#         print(json.dumps(result, indent=2))



"""
Unified PDF Extractor (Parts 1–6)
---------------------------------
- Part 1: Text extraction with OCR fallback
- Part 2: Table extraction (Camelot + pdfplumber, scored)
- Part 3: Layout extraction (PyMuPDF: text, spans, fonts, images, tables)
- Part 5: Metadata & provenance tagging (JSONL, per-page JSON/MD, global Markdown)
- Part 6: Storage formats (Markdown, JSON, TXT exports)
"""

import json
import re
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import camelot
import pandas as pd
import fitz  # PyMuPDF
from pathlib import Path
from ocr_config import test_ocr_setup
import time, psutil, os

# Optional managed service import (DocAI)
try:
    from docai_integration import process_docai
except ImportError:
    process_docai = None
    
# Ensure consistent project paths
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PARSED_DIR = BASE_DIR / "data" / "parsed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PARSED_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Helpers
# =====================================================
def int_to_rgb(color_int):
    if color_int is None:
        return (0, 0, 0)
    r = (color_int >> 16) & 255
    g = (color_int >> 8) & 255
    b = color_int & 255
    return (r, g, b)


def rect_to_list(rect):
    if rect is None:
        return []
    return [rect.x0, rect.y0, rect.x1, rect.y1]


def decode_font_flags(flags: int):
    return {
        "superscript": bool(flags & 1),
        "italic": bool(flags & 2),
        "serif": bool(flags & 4),
        "monospace": bool(flags & 8),
        "bold": bool(flags & 16),
    }


def classify_block(block, page_height):
    text = " ".join(
        span["text"] for line in block["lines"] for span in line["spans"]
    ).strip()
    if not text:
        return None, None
    sizes = [span["size"] for line in block["lines"] for span in line["spans"]]
    max_size = max(sizes)
    avg_size = sum(sizes) / len(sizes)
    x0, y0, x1, y1 = block["bbox"]
    if y1 < page_height * 0.1:
        return "header", text
    if y0 > page_height * 0.9:
        return "footer", text
    if max_size >= avg_size * 1.5:
        return "title", text
    if max_size >= avg_size * 1.2:
        return "heading", text
    return "paragraph", text


def score_table(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return -1
    rows, cols = df.shape
    if rows < 3 or cols < 2:
        return -1
    numeric_cells = df.applymap(
        lambda x: str(x).replace(",", "").replace(".", "").isdigit()
    ).sum().sum()
    numeric_ratio = numeric_cells / (rows * cols)
    return (rows * 0.1) + (cols * 0.2) + (numeric_ratio * 2)


def parse_filename(doc_id: str):
    match = re.match(r"([A-Za-z0-9]+).*?(\d{4})", doc_id)
    if match:
        company, year = match.groups()
        return company, int(year)
    return doc_id, None


# =====================================================
# Unified Runner (Parts 1–6)
# =====================================================
def process_pdf(pdf_path, output_dir=PARSED_DIR):
    pdf_path = Path(pdf_path)
    pdf_name = pdf_path.stem
    output_path = Path(output_dir) / pdf_name
    (output_path / "text").mkdir(parents=True, exist_ok=True)
    (output_path / "tables").mkdir(parents=True, exist_ok=True)
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "json").mkdir(parents=True, exist_ok=True)
    (output_path / "metadata").mkdir(parents=True, exist_ok=True)

    # OCR setup
    ocr_ok, ocr_msg = test_ocr_setup()
    ocr_available = ocr_ok

    doc = fitz.open(pdf_path)
    global_layout = []
    ocr_pages = []
    word_boxes = []
    table_metadata = {}
    saved_tables = 0

    # ---- Part 3: Layout detection first ----
    for page_num, page in enumerate(doc, 1):
        page_height = page.rect.height
        page_layout = {
            "page": page_num,
            "text_file": f"text/page{page_num:03d}.txt",
            "blocks": []
        }

        # text dict (with spans/fonts)
        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            if "lines" not in block:
                continue

            block_lines = []
            for line in block["lines"]:
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                if not line_text:
                    continue
                block_lines.append({
                    "text": line_text,
                    "spans": [
                        {
                            "text": span["text"],
                            "font": span["font"],
                            "size": span["size"],
                            "color": int_to_rgb(span.get("color", 0)),
                            "style": decode_font_flags(span["flags"])
                        }
                        for span in line["spans"]
                    ]
                })

            if block_lines:
                block_type, _ = classify_block(block, page_height)
                page_layout["blocks"].append({
                    "type": block_type or "paragraph",
                    "bbox": list(block["bbox"]),
                    "lines": block_lines
                })

        # ---- Part 1: Text extraction ----
        with pdfplumber.open(pdf_path) as pdf:
            pl_page = pdf.pages[page_num - 1]
            text = pl_page.extract_text(x_density=2, y_density=2)
            words = pl_page.extract_words() or []
            if text and text.strip():
                (output_path / f"text/page{page_num:03d}.txt").write_text(text, encoding="utf-8")
            else:
                if ocr_available:
                    images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
                    if images:
                        text = pytesseract.image_to_string(images[0]).strip()
                        (output_path / f"text/page{page_num:03d}.txt").write_text(text, encoding="utf-8")
                        ocr_pages.append(page_num)
            if words:
                word_boxes.append({"page_num": page_num, "words": words})

        # ---- Part 2: Table extraction ----
        candidates = []
        try:
            tables = camelot.read_pdf(str(pdf_path), pages=str(page_num), flavor="stream")
            for t in tables:
                if not t.df.empty:
                    candidates.append(("camelot_stream", t.df, list(t._bbox)))
        except Exception:
            pass
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pl_page = pdf.pages[page_num - 1]
                for table in pl_page.extract_tables() or []:
                    if table and len(table) >= 2 and len(table[0]) >= 2:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        candidates.append(("pdfplumber", df, list(pl_page.bbox)))
        except Exception:
            pass

        scored = []
        for method, df, bbox in candidates:
            s = score_table(df)
            if s > 0:
                scored.append((s, method, df, bbox))

        if scored:
            best_score, best_method, best_df, best_bbox = max(scored, key=lambda x: x[0])
            saved_tables += 1
            filename = f"page{page_num:03d}_best.csv"
            filepath = output_path / "tables" / filename
            best_df.to_csv(filepath, index=False, encoding="utf-8-sig")
            table_metadata[page_num] = {
                "csv": f"tables/{filename}",
                "bbox": best_bbox,
                "rows": best_df.shape[0],
                "cols": best_df.shape[1],
                "method": best_method,
                "score": best_score
            }
            page_layout["blocks"].append({
                "type": "table",
                **table_metadata[page_num]
            })

        # ---- Images ----
        for img_idx, img in enumerate(page.get_images(full=True), 1):
            xref = img[0]
            bbox = page.get_image_bbox(img)
            img_path = output_path / "images" / f"page{page_num:03d}_img{img_idx}.png"
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha < 4:
                pix.save(img_path)
            else:
                fitz.Pixmap(fitz.csRGB, pix).save(img_path)
            page_layout["blocks"].append({
                "type": "image",
                "bbox": rect_to_list(bbox),
                "path": f"images/{img_path.name}"
            })

        # save per-page JSON (Part 3)
        (output_path / "json" / f"page{page_num:03d}.json").write_text(
            json.dumps(page_layout, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        global_layout.append(page_layout)

    # ---- Save global layout ----
    (output_path / "layout.json").write_text(
        json.dumps(global_layout, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- Logs ----
    (output_path / "ocr_log.json").write_text(json.dumps({
        "pdf_path": str(pdf_path),
        "pages_requiring_ocr": ocr_pages,
        "ocr_page_count": len(ocr_pages),
        "total_pages": len(doc)
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    (output_path / "word_boxes.json").write_text(json.dumps(word_boxes, indent=2, ensure_ascii=False), encoding="utf-8")

    (output_path / "tables" / "tables_metadata.json").write_text(
        json.dumps(table_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- Part 5: Metadata & provenance ----
    company, fiscal_year = parse_filename(pdf_name)

    metadata_dir = output_path / "metadata"
    jsonl_path = metadata_dir / "metadata_provenance.jsonl"
    md_path = metadata_dir / "metadata_provenance.md"

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for page in global_layout:
            page_num = page["page"]
            page_records = []
            page_md_lines = [f"# Page {page_num}", ""]
            current_section = "UNLABELED"

            for block in page.get("blocks", []):
                if block["type"] in ("title", "heading"):
                    current_section = block["lines"][0]["text"] if block.get("lines") else block.get("text", "")

                record = {
                    "doc_id": pdf_name,
                    "company": company,
                    "fiscal_year": fiscal_year,
                    "page": page_num,
                    "section": current_section,
                    "block_type": block.get("type"),
                    "bbox": block.get("bbox", []),
                    "text": "\n".join(line["text"] for line in block.get("lines", [])) if block.get("lines") else block.get("text", ""),
                    "source_path": None,
                }

                if block["type"] == "table":
                    record["source_path"] = block.get("csv")
                    page_md_lines.append(f"[Table: {block['csv']} | rows={block.get('rows')}, cols={block.get('cols')}]")
                elif block["type"] == "image":
                    record["source_path"] = block.get("path")
                    page_md_lines.append(f"![Image]({block['path']})")
                elif block["type"] in ("title", "heading"):
                    record["source_path"] = page.get("text_file")
                    page_md_lines.append(f"## {record['text']}")
                elif block["type"] == "paragraph":
                    record["source_path"] = page.get("text_file")
                    for line in block.get("lines", []):
                        page_md_lines.append(line["text"])

                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
                page_records.append(record)

            (metadata_dir / f"metadata_provenance_page{page_num:03d}.json").write_text(
                json.dumps(page_records, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            (metadata_dir / f"metadata_provenance_page{page_num:03d}.md").write_text(
                "\n".join(page_md_lines),
                encoding="utf-8"
            )

    # ---- Global provenance Markdown ----
    md_lines = [f"# Metadata Knowledge Base: {pdf_name}", ""]
    for page in global_layout:
        for block in page.get("blocks", []):
            if block["type"] in ("title", "heading"):
                md_lines.append(f"## {block['lines'][0]['text'] if block.get('lines') else block.get('text','')}")
            elif block["type"] == "paragraph":
                for line in block.get("lines", []):
                    md_lines.append(line["text"])
            elif block["type"] == "table":
                md_lines.append(f"[Table: {block['csv']} | rows={block['rows']}, cols={block['cols']}]")
            elif block["type"] == "image":
                md_lines.append(f"![Image]({block['path']})")
        md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # ---- Part 6: Formats ----
    (output_path / f"{pdf_name}.md").write_text("\n".join(md_lines), encoding="utf-8")
    (output_path / f"{pdf_name}.json").write_text(
        json.dumps(global_layout, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_path / f"{pdf_name}.txt").write_text(
        "\n".join(
            line["text"]
            for page in global_layout
            for block in page["blocks"]
            if block["type"] in ("title", "heading", "paragraph")
            for line in block.get("lines", [])
        ),
        encoding="utf-8"
    )

    return {
        "pdf": pdf_name,
        "pages": len(global_layout),
        "ocr_pages": len(ocr_pages),
        "tables": saved_tables,
        "output_dir": str(output_path)
    }

if __name__ == "__main__":
    PROJECT_ID = "docai-test-473000"
    LOCATION = "us"
    PROCESSOR_ID = "cd1280bfb6a64860"

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    summaries = []

    for pdf_file in pdf_files:
        print(f"\n=== Processing {pdf_file.name} with BOTH pipelines ===")

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        start = time.time()

        summary = {"pdf": pdf_file.stem, "output_dir": str(PARSED_DIR / pdf_file.stem)}

        try:
            # ---- Always run open-source pipeline ----
            open_source_summary = process_pdf(pdf_file)
            summary["open_source"] = open_source_summary

            # ---- Always run DocAI pipeline ----
            if process_docai:
                try:
                    docai_summary = process_docai(
                        pdf_file,
                        PARSED_DIR,
                        project_id=PROJECT_ID,
                        location=LOCATION,
                        processor_id=PROCESSOR_ID
                    )
                    summary["docai"] = docai_summary
                except Exception as e:
                    summary["docai_error"] = str(e)
            else:
                summary["docai_error"] = "docai_integration not available"

            # ---- Runtime + memory ----
            duration = time.time() - start
            mem_after = process.memory_info().rss / (1024 * 1024)
            peak_mem = max(mem_before, mem_after)

            summary.update({
                "runtime_sec": round(duration, 2),
                "memory_mb": round(peak_mem, 2)
            })

            summaries.append(summary)

            print(f"Done: {summary['pdf']} "
                  f"({summary['open_source']['pages']} pages, "
                  f"{summary['open_source']['tables']} tables, "
                  f"time={summary['runtime_sec']}s, "
                  f"mem={summary['memory_mb']}MB)")
        except Exception as e:
            print(f"Failed on {pdf_file.name}: {e}")

    # ---- Save combined summary ----
    summary_path = PARSED_DIR / "summary_dual.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nAll done. Results saved to {summary_path}")


