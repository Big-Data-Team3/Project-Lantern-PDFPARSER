from google.cloud import documentai_v1 as documentai
from pathlib import Path
import os, csv, json
from PyPDF2 import PdfReader, PdfWriter

# Point to your service account JSON
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    str(Path(__file__).resolve().parent.parent / "secrets/gcloud-key.json")
)


def extract_pages(pdf_path, out_path, start=1, end=15, max_pages=15):
    """
    Extract pages from 'start' to 'end' (1-based inclusive), but capped at max_pages.
    Returns the new PDF path, number of pages, and the actual start page.
    """
    reader = PdfReader(pdf_path)
    writer = PdfWriter()

    end = min(end, start + max_pages - 1, len(reader.pages))
    for i in range(start - 1, end):
        writer.add_page(reader.pages[i])

    with open(out_path, "wb") as f:
        writer.write(f)

    print(f"[DocAI] Extracted pages {start}–{end} → {out_path}")
    return out_path, (end - start + 1), start


def analyze_pdf_with_docai(pdf_path, project_id, location, processor_id):
    """Send a PDF to Google Document AI for OCR + table detection."""
    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(project_id, location, processor_id)

    pdf_bytes = Path(pdf_path).read_bytes()
    raw_document = documentai.RawDocument(content=pdf_bytes, mime_type="application/pdf")

    request = documentai.ProcessRequest(name=name, raw_document=raw_document)
    result = client.process_document(request=request)
    return result.document


def save_results(doc, output_dir, start_page=1, batch_tag=None):
    """Save JSON, text, tables, and metadata mapped to original page numbers."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw JSON
    raw_json_path = output_dir / f"docai_output{f'_{batch_tag}' if batch_tag else ''}.json"
    raw_json_path.write_text(str(doc), encoding="utf-8")

    # Save full text
    text_path = output_dir / f"docai_text{f'_{batch_tag}' if batch_tag else ''}.txt"
    text_path.write_text(doc.text, encoding="utf-8")

    print(f"[DocAI] Saved raw JSON → {raw_json_path}")
    print(f"[DocAI] Saved text file → {text_path}")

    metadata = {}
    table_count = 0

    for page_index, page in enumerate(doc.pages, 0):
        actual_page = start_page + page_index
        page_tables = []

        for ti, table in enumerate(page.tables, 1):
            table_count += 1
            filename = f"page{actual_page:03d}_table{ti}.csv"
            table_path = output_dir / filename

            with open(table_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                def get_text(layout):
                    if not layout.text_anchor.text_segments:
                        return ""
                    s = ""
                    for seg in layout.text_anchor.text_segments:
                        start = int(seg.start_index) if seg.start_index else 0
                        end = int(seg.end_index)
                        s += doc.text[start:end]
                    return s.strip()

                for row in list(table.header_rows) + list(table.body_rows):
                    cells = [get_text(cell.layout) for cell in row.cells]
                    writer.writerow(cells)

            print(f"[DocAI] Saved table → {table_path}")
            page_tables.append({
                "csv": str(table_path),
                "table_index": ti,
                "rows": len(table.body_rows) + len(table.header_rows),
                "cols": max(len(r.cells) for r in list(table.header_rows) + list(table.body_rows))
                        if (list(table.header_rows) + list(table.body_rows)) else 0
            })

        if page_tables:
            metadata[actual_page] = page_tables

    # Save metadata JSON for this batch
    meta_path = output_dir / f"tables_metadata{f'_{batch_tag}' if batch_tag else ''}.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[DocAI] Saved table metadata → {meta_path}")

    return table_count, metadata


def process_docai(pdf_path, output_dir, project_id, location, processor_id,
                  start_page=1, end_page=None, batch_size=15):
    """
    Splits PDF into ≤batch_size chunks (default 15),
    sends each chunk to DocAI, saves results, merges metadata.
    """
    pdf_name = Path(pdf_path).stem
    docai_dir = Path(output_dir) / pdf_name / "docai"
    docai_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    if end_page is None:
        end_page = total_pages

    all_metadata = {}
    total_tables = 0
    processed_batches = 0

    for start in range(start_page, end_page + 1, batch_size):
        stop = min(start + batch_size - 1, end_page)

        # Subset PDF for DocAI
        subset_pdf = docai_dir / f"docai_subset_{start:03d}-{stop:03d}.pdf"
        subset_pdf, pages_extracted, actual_start = extract_pages(
            pdf_path, subset_pdf, start=start, end=stop, max_pages=batch_size
        )

        print(f"[DocAI] Processing {subset_pdf} with Google Document AI...")

        # Run through DocAI
        doc = analyze_pdf_with_docai(subset_pdf, project_id, location, processor_id)

        # Save results for this batch
        batch_tag = f"{start:03d}-{stop:03d}"
        table_count, metadata = save_results(doc, docai_dir, start_page=actual_start, batch_tag=batch_tag)
        total_tables += table_count
        all_metadata.update(metadata)
        processed_batches += 1

    # Save merged metadata
    merged_meta_path = docai_dir / "tables_metadata_merged.json"
    merged_meta_path.write_text(json.dumps(all_metadata, indent=2), encoding="utf-8")
    print(f"[DocAI] Saved merged table metadata → {merged_meta_path}")

    return {
        "docai_batches": processed_batches,
        "docai_pages": end_page - start_page + 1,
        "docai_tables": total_tables,
        "docai_output_dir": str(docai_dir),
        "merged_metadata": str(merged_meta_path)
    }


if __name__ == "__main__":
    # Example run (replace IDs with your own)
    pdf_path = "../data/raw/NVIDIA_10-K_2024-02-21.pdf"
    output_dir = "../data/parsed"
    project_id = "docai-test-473000"
    location = "us"
    processor_id = "cd1280bfb6a64860"

    summary = process_docai(pdf_path, output_dir, project_id, location, processor_id)
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
