"""
Part 3 — Full Layout Extraction (PyMuPDF)
- Per-page JSON + global layout.json
- Extracts text with font, size, color, style
- Detects headers, footers, titles, paragraphs
- Detects list items (inside paragraphs)
- Saves images + links + table references
- Converts Rects to JSON-safe lists
- Decodes font flags into human-readable styles
"""

import fitz  # PyMuPDF
from pathlib import Path
import json
import re


# ---------- Helpers ----------
def int_to_rgb(color_int):
    """Convert PyMuPDF color integer to RGB tuple."""
    if color_int is None:
        return (0, 0, 0)
    r = (color_int >> 16) & 255
    g = (color_int >> 8) & 255
    b = color_int & 255
    return (r, g, b)


def rect_to_list(rect):
    """Convert a PyMuPDF Rect to a plain list."""
    if rect is None:
        return []
    return [rect.x0, rect.y0, rect.x1, rect.y1]


def decode_font_flags(flags: int):
    """Decode PyMuPDF font flags into human-readable styles."""
    return {
        "superscript": bool(flags & 1),
        "italic": bool(flags & 2),
        "serif": bool(flags & 4),
        "monospace": bool(flags & 8),
        "bold": bool(flags & 16),
    }


def classify_block(block, page_height):
    """Heuristics for block classification."""
    text = " ".join(span["text"] for line in block["lines"] for span in line["spans"]).strip()
    if not text:
        return None, None

    sizes = [span["size"] for line in block["lines"] for span in line["spans"]]
    max_size = max(sizes)
    avg_size = sum(sizes) / len(sizes)

    x0, y0, x1, y1 = block["bbox"]

    # Detect headers / footers
    if y1 < page_height * 0.1:
        return "header", text
    if y0 > page_height * 0.9:
        return "footer", text

    # Detect titles / headings
    if max_size >= avg_size * 1.5:
        return "title", text
    if max_size >= avg_size * 1.2:
        return "heading", text

    return "paragraph", text


def detect_list_items(text):
    """Detect bullet or numbered list items inside text."""
    list_pattern = re.compile(r"^(?:[-•*]|\d+\.)\s+")
    items = []
    for line in text.split("\n"):
        if list_pattern.match(line.strip()):
            items.append(line.strip())
    return items


# ---------- Main Extraction ----------
def extract_layout(pdf_path, output_dir="../data/parsed", dpi=300):
    pdf_name = Path(pdf_path).stem
    output_path = Path(output_dir) / pdf_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Create images folder
    img_dir = output_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    global_layout = []

    for page_num, page in enumerate(doc, 1):
        page_height = page.rect.height
        page_layout = {"page": page_num, "blocks": []}

        # ---- Text blocks ----
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

            block_entry = {
                "type": block_type,
                "text": block_text,
                "bbox": list(block["bbox"]),
                "spans": spans
            }
            if list_items:
                block_entry["list_items"] = list_items

            page_layout["blocks"].append(block_entry)

        # ---- Links ----
        for link in page.get_links():
            if "uri" in link:
                page_layout["blocks"].append({
                    "type": "link",
                    "uri": link["uri"],
                    "bbox": rect_to_list(link.get("from"))
                })

        # ---- Images ----
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
                "path": str(img_path)
            })

        # ---- Lines & Shapes ----
        for d in page.get_drawings():
            if d["type"] == "line":
                page_layout["blocks"].append({
                    "type": "line",
                    "bbox": rect_to_list(d["rect"]),
                    "color": d["color"]
                })

        # ---- Table references ----
        table_ref_path = output_path / "tables" / f"page{page_num:03d}_best.csv"
        if table_ref_path.exists():
            page_layout["blocks"].append({
                "type": "table_ref",
                "page": page_num,
                "csv": str(table_ref_path)
            })

        # Save per-page JSON
        with open(output_path / f"page{page_num:03d}.json", "w", encoding="utf-8") as f:
            json.dump(page_layout, f, indent=2)

        global_layout.append(page_layout)

    # Save global layout.json
    with open(output_path / "layout.json", "w", encoding="utf-8") as f:
        json.dump(global_layout, f, indent=2)

    print(f"[LAYOUT] {pdf_name}: {len(global_layout)} pages exported with font styles decoded")
    return global_layout


# ---------- Runner ----------
if __name__ == "__main__":
    pdf_files = list(Path("../data/raw").glob("NVIDIA_*.pdf"))
    for pdf_file in pdf_files:
        print(f"\nProcessing layout: {pdf_file.name}")
        extract_layout(pdf_file)
