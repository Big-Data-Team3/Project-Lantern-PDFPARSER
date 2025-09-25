import logging
import time
import json
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from rapidocr_onnxruntime import RapidOCR

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem
import platform


def get_poppler_path():
    if platform.system() == "Windows":
        candidates = [
            r"C:\poppler-25.07.0\Library\bin",  # your install
            r"C:\Program Files\poppler\bin",
            r"C:\Program Files (x86)\poppler\bin",
        ]
        for c in candidates:
            if Path(c).exists():
                return c
        return None
    else:
        return None


def main():
    # === Directory setup ===
    base_dir = Path("../data/docling")
    base_dir.mkdir(parents=True, exist_ok=True)

    # === Logging setup ===
    log_file = base_dir / "docling_extraction.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="w", encoding="utf-8")
        ]
    )
    log = logging.getLogger("docling_rapidocr_extraction")

    input_pdfs = sorted(Path("../data/raw").glob("*.pdf"))
    if not input_pdfs:
        log.error("No PDFs found in data/raw/")
        return
    input_pdf = input_pdfs[0]
    doc_stem = input_pdf.stem

    log.info(f"Input: {input_pdf.name}")
    start_total = time.time()
    log.info("Starting Docling conversion with RapidOCR fallback...")

    # === Docling setup ===
    pipe_opts = PdfPipelineOptions(
        force_ocr=False,
        ocr_engine="none",
        images_scale=2.0,
        generate_picture_images=True,
        generate_page_images=False
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipe_opts)}
    )
    start = time.time()
    conv_res = converter.convert(input_pdf)
    doc = conv_res.document
    log.info(f"Conversion complete in {time.time() - start:.2f} seconds.")

    # === Subdirs ===
    images_dir = base_dir / "images" / doc_stem
    tables_dir = base_dir / "tables" / doc_stem
    text_dir = base_dir / "text" / doc_stem
    json_dir = base_dir / "json" / doc_stem
    md_dir = base_dir / "markdown" / doc_stem
    for d in [images_dir, tables_dir, text_dir, json_dir, md_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # === Save full JSON ===
    full_json_path = json_dir / f"{doc_stem}.json"
    with open(full_json_path, "w", encoding="utf-8") as f:
        json.dump(doc.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
    log.info(f"Full document JSON → {full_json_path}")

    # === Save full Markdown ===
    full_md_path = md_dir / f"{doc_stem}.md"
    with open(full_md_path, "w", encoding="utf-8") as f:
        f.write(doc.export_to_markdown())
    log.info(f"Full document Markdown → {full_md_path}")

    # === Save full Text (clean, from dict) ===
    full_txt_path = text_dir / f"{doc_stem}.txt"
    try:
        doc_dict = doc.export_to_dict()
        text_blocks = []
        for page in doc_dict.get("pages", []):
            if isinstance(page, dict):
                for block in page.get("blocks", []):
                    if isinstance(block, dict) and block.get("category") == "text" and block.get("text"):
                        text_blocks.append(block["text"])
        with open(full_txt_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(text_blocks))
        log.info(f"Full document Text → {full_txt_path}")
    except Exception as e:
        log.warning(f"Failed to save full Text: {e}")

    # === Save Images ===
    picture_count = 0
    for element, _level in doc.iterate_items():
        if isinstance(element, PictureItem):
            img = element.get_image(doc)
            out_path = images_dir / f"{doc_stem}-picture-{picture_count+1:03}.png"
            img.save(out_path, "PNG")
            log.info(f"Saved image → {out_path}")
            picture_count += 1
    if picture_count == 0:
        log.warning("No images were extracted.")

    # === Save Tables ===
    table_count = 0
    for element, _level in doc.iterate_items():
        if isinstance(element, TableItem):
            df = element.export_to_dataframe(doc=doc)
            out_path = tables_dir / f"{doc_stem}-table-{table_count+1:03}.csv"
            df.to_csv(out_path, index=False)
            log.info(f"Saved table → {out_path}")
            table_count += 1
    if table_count == 0:
        log.warning("No tables were extracted.")

    # === Load all pages as images once (for OCR fallback) ===
    page_images = convert_from_path(str(input_pdf), dpi=300, poppler_path=get_poppler_path())

    # === Load RapidOCR ===
    ocr = RapidOCR()

    # === Per-page processing ===
    total_pages = len(doc.pages)
    for page in doc.pages.values():
        page_no = page.page_no
        start_page = time.time()
        log.info(f"Processing page {page_no}/{total_pages}...")

        txt_path = text_dir / f"{doc_stem}-page-{page_no:03}.txt"
        md_path = md_dir / f"{doc_stem}-page-{page_no:03}.md"
        json_path = json_dir / f"{doc_stem}-page-{page_no:03}.json"

        blocks = getattr(page, "blocks", [])
        extracted_text = ""
        try:
            extracted_text = "\n".join(b.text for b in blocks if hasattr(b, "text"))
        except Exception:
            extracted_text = ""

        # === Save native Docling text or fallback ===
        if extracted_text.strip():
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(f"### Block {i+1}\n\n{b.text}"
                                    for i, b in enumerate(blocks) if hasattr(b, "text")))
            log.info(f"Page {page_no:03}: Extracted via Docling → {txt_path}")
        else:
            try:
                np_image = np.array(page_images[page_no - 1])
                results, _ = ocr(np_image)
                ocr_text = "\n".join([line[1] for line in results]).strip()

                if ocr_text:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(ocr_text)
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(f"### OCR Extracted Page {page_no}\n\n{ocr_text}")
                    log.info(f"Page {page_no:03}: Extracted via RapidOCR → {txt_path}")
                else:
                    log.warning(f"Page {page_no:03}: No text extracted even with OCR")
            except Exception as e:
                log.warning(f"Page {page_no:03}: OCR fallback failed - {e}")

        # Save JSON per page
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(page.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.warning(f"Page {page_no:03}: Failed to save page JSON - {e}")

        log.info(f"Page {page_no:03} processed in {time.time() - start_page:.2f} sec")

    log.info(f"All extraction done in {time.time() - start_total:.2f} sec (see log file at {log_file})")


if __name__ == "__main__":
    main()
