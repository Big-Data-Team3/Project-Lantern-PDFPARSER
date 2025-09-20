"""
Unified PDF Extractor (Parts 1, 2, 3, 5)
----------------------------------------
- Part 1: Text extraction with OCR fallback
- Part 2: Table extraction (Camelot + pdfplumber)
- Part 3: Layout extraction (PyMuPDF: text, fonts, spans, lists, images, tables)
- Part 5: Metadata & provenance tagging (JSONL, per-page JSON/MD, global Markdown)
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
    text = " ".join(span["text"] for line in block["lines"] for span in line["spans"]).strip()
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


def detect_list_items(text):
    import re
    list_pattern = re.compile(r"^(?:[-•*]|\d+\.)\s+")
    return [line.strip() for line in text.split("\n") if list_pattern.match(line.strip())]


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
# Part 1 - Text Extraction
# =====================================================
class PDFTextExtractor:
    def __init__(self):
        ocr_ok, ocr_msg = test_ocr_setup()
        if ocr_ok:
            print(f"OCR ready: {ocr_msg}")
            self.ocr_available = True
        else:
            print(f"WARNING: {ocr_msg}")
            self.ocr_available = False

    def extract_text_page(self, pdf_path, page_num, text_dir):
        page_file = text_dir / f"page{page_num:03d}.txt"
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num - 1]
            text = page.extract_text(x_density=2, y_density=2)
            words = page.extract_words() or []

            if text and text.strip():
                page_file.write_text(text, encoding="utf-8")
                return text, words, False
            else:
                if self.ocr_available:
                    images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
                    if images:
                        text = pytesseract.image_to_string(images[0]).strip()
                        page_file.write_text(text, encoding="utf-8")
                        return text, [], True
        return None, [], False

    def process_pdf(self, pdf_path, output_path):
        text_dir = output_path / "text"
        text_dir.mkdir(parents=True, exist_ok=True)

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

        ocr_pages = []
        word_boxes = []

        for page_num in range(1, total_pages + 1):
            text, words, used_ocr = self.extract_text_page(pdf_path, page_num, text_dir)
            if used_ocr:
                ocr_pages.append(page_num)
            if words:
                word_boxes.append({"page_num": page_num, "words": words})

        (output_path / "ocr_log.json").write_text(json.dumps({
            "pdf_path": str(pdf_path),
            "pages_requiring_ocr": ocr_pages,
            "ocr_page_count": len(ocr_pages),
            "total_pages": total_pages
        }, indent=2), encoding="utf-8")

        (output_path / "word_boxes.json").write_text(json.dumps(word_boxes, indent=2), encoding="utf-8")

        print(f"[TEXT] {pdf_path.name}: {total_pages} pages, {len(ocr_pages)} with OCR")
        return total_pages, len(ocr_pages)


# =====================================================
# Part 2 - Table Extraction
# =====================================================
def extract_camelot(pdf_path, page_num):
    results = []
    try:
        tables = camelot.read_pdf(str(pdf_path), pages=str(page_num), flavor="stream")
        for t in tables:
            if not t.df.empty:
                results.append(("camelot_stream", t.df, list(t._bbox)))
    except Exception:
        pass
    return results


def extract_pdfplumber(pdf_path, page_num):
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num - 1]
            for table in page.extract_tables() or []:
                if table and len(table) >= 2 and len(table[0]) >= 2:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    results.append(("pdfplumber", df, list(page.bbox)))
    except Exception:
        pass
    return results


def process_tables(pdf_path, output_path):
    table_dir = output_path / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)

    metadata = {}
    saved_tables = 0

    for page_num in range(1, num_pages + 1):
        candidates = []
        candidates += extract_camelot(pdf_path, page_num)
        candidates += extract_pdfplumber(pdf_path, page_num)

        if not candidates:
            continue

        scored = []
        for method, df, bbox in candidates:
            if df.empty:
                continue
            score = score_table(df)
            if score <= 0:
                continue
            scored.append((score, method, df, bbox))

        if not scored:
            continue

        best_score, best_method, best_df, best_bbox = max(scored, key=lambda x: x[0])
        saved_tables += 1

        filename = f"page{page_num:03d}_best.csv"
        filepath = table_dir / filename
        best_df.to_csv(filepath, index=False)

        metadata[page_num] = {
            "csv": f"tables/{filename}",
            "bbox": best_bbox,
            "rows": best_df.shape[0],
            "cols": best_df.shape[1],
            "method": best_method,
            "score": best_score,
        }

    (table_dir / "tables_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[TABLES] {pdf_path.name}: {saved_tables} tables saved")
    return metadata


# =====================================================
# Part 3 - Layout Extraction
# =====================================================
def extract_layout(pdf_path, output_path, table_metadata):
    img_dir = output_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    json_dir = output_path / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    global_layout = []

    for page_num, page in enumerate(doc, 1):
        page_height = page.rect.height
        page_layout = {
            "page": page_num,
            "text_file": f"text/page{page_num:03d}.txt",
            "blocks": []
        }

        text_dict = page.get_text("dict")
        for block in text_dict["blocks"]:
            if "lines" not in block:
                continue
            block_type, block_text = classify_block(block, page_height)
            if not block_text:
                continue
            spans = []
            for line in block["lines"]:
                for span in line["spans"]:
                    spans.append({
                        "text": span["text"],
                        "font": span["font"],
                        "size": span["size"],
                        "color": int_to_rgb(span.get("color", 0)),
                        "style": decode_font_flags(span["flags"])
                    })
            list_items = detect_list_items(block_text) if block_type == "paragraph" else []
            block_entry = {"type": block_type, "text": block_text, "bbox": list(block["bbox"]), "spans": spans}
            if list_items:
                block_entry["list_items"] = list_items
            page_layout["blocks"].append(block_entry)

        for link in page.get_links():
            if "uri" in link:
                page_layout["blocks"].append({
                    "type": "link",
                    "uri": link["uri"],
                    "bbox": rect_to_list(link.get("from"))
                })

        for img_idx, img in enumerate(page.get_images(full=True), 1):
            xref = img[0]
            bbox = page.get_image_bbox(img)
            img_path = img_dir / f"page{page_num:03d}_img{img_idx}.png"
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha < 4:
                pix.save(img_path)
            else:
                pix_converted = fitz.Pixmap(fitz.csRGB, pix)
                pix_converted.save(img_path)
                pix_converted = None
            pix = None
            page_layout["blocks"].append({
                "type": "image",
                "bbox": rect_to_list(bbox),
                "path": f"images/{img_path.name}"
            })

        if page_num in table_metadata:
            meta = table_metadata[page_num]
            page_layout["blocks"].append({
                "type": "table",
                "csv": meta["csv"],
                "bbox": meta["bbox"],
                "rows": meta["rows"],
                "cols": meta["cols"],
                "method": meta["method"],
                "score": meta["score"]
            })

        (json_dir / f"page{page_num:03d}.json").write_text(json.dumps(page_layout, indent=2), encoding="utf-8")
        global_layout.append(page_layout)

    (output_path / "layout.json").write_text(json.dumps(global_layout, indent=2), encoding="utf-8")

    print(f"[LAYOUT] {pdf_path.name}: {len(global_layout)} pages exported (with table metadata)")
    return global_layout


# =====================================================
# Part 5 - Metadata & Provenance
# =====================================================
def build_metadata(parsed_dir):
    parsed_dir = Path(parsed_dir)
    layout_path = parsed_dir / "layout.json"
    if not layout_path.exists():
        raise FileNotFoundError(f"layout.json not found in {parsed_dir}")

    layout = json.loads(layout_path.read_text())
    doc_id = parsed_dir.name
    company, fiscal_year = parse_filename(doc_id)

    metadata_dir = parsed_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = metadata_dir / "metadata_provenance.jsonl"
    md_path = metadata_dir / "metadata_provenance.md"

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for page in layout:
            page_num = page["page"]
            page_records = []
            page_md_lines = [f"# Page {page_num}", ""]
            current_section = "UNLABELED"

            for block in page.get("blocks", []):
                if block["type"] in ("title", "heading"):
                    current_section = block["text"]

                record = {
                    "doc_id": doc_id,
                    "company": company,
                    "fiscal_year": fiscal_year,
                    "page": page_num,
                    "section": current_section,
                    "block_type": block.get("type"),
                    "bbox": block.get("bbox", []),
                    "text": block.get("text", ""),
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
                    page_md_lines.append(f"## {block['text']}")
                elif block["type"] == "paragraph":
                    record["source_path"] = page.get("text_file")
                    page_md_lines.append(block["text"])
                elif block["type"] == "list":
                    record["source_path"] = page.get("text_file")
                    for item in block.get("list_items", []):
                        page_md_lines.append(f"- {item}")

                jf.write(json.dumps(record) + "\n")
                page_records.append(record)

            page_json_path = metadata_dir / f"metadata_provenance_page{page_num:03d}.json"
            page_json_path.write_text(json.dumps(page_records, indent=2), encoding="utf-8")

            page_md_path = metadata_dir / f"metadata_provenance_page{page_num:03d}.md"
            page_md_path.write_text("\n".join(page_md_lines), encoding="utf-8")

    print(f"JSONL knowledge base saved → {jsonl_path}")
    print(f"Per-page JSON + Markdown files saved in → {metadata_dir}")

    md_lines = [f"# Metadata Knowledge Base: {doc_id}", ""]
    current_section = None
    for page in layout:
        for block in page.get("blocks", []):
            if block["type"] in ("title", "heading"):
                current_section = block["text"]
                md_lines.append(f"## {current_section}")
            elif block["type"] == "paragraph":
                md_lines.append(block["text"])
            elif block["type"] == "list":
                for item in block.get("list_items", []):
                    md_lines.append(f"- {item}")
            elif block["type"] == "table":
                md_lines.append(f"[Table: {block['csv']} | rows={block.get('rows')}, cols={block.get('cols')}]")
            elif block["type"] == "image":
                md_lines.append(f"![Image]({block['path']})")
        md_lines.append("")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[PART 5] Global Markdown knowledge base saved → {md_path}")

    return jsonl_path, md_path, metadata_dir


# =====================================================
# Unified Runner
# =====================================================
def process_pdf(pdf_path, output_dir="../data/parsed"):
    pdf_name = Path(pdf_path).stem
    output_path = Path(output_dir) / pdf_name
    output_path.mkdir(parents=True, exist_ok=True)

    extractor = PDFTextExtractor()
    pages, ocr_count = extractor.process_pdf(pdf_path, output_path)

    table_metadata = process_tables(pdf_path, output_path)

    layout = extract_layout(pdf_path, output_path, table_metadata)

    build_metadata(output_path)

    return {
        "pages": pages,
        "ocr_pages": ocr_count,
        "tables": len(table_metadata),
        "layout_pages": len(layout)
    }


if __name__ == "__main__":
    raw_dir = Path("../data/raw")
    parsed_dir = Path("../data/parsed")

    pdf_files = list(raw_dir.glob("NVIDIA_*.pdf"))
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        summary = process_pdf(pdf_file, parsed_dir)
        print(f"Summary: {summary}")

