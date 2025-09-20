try:
    import pdfplumber
    print("pdfplumber: OK")
except ImportError:
    print("pdfplumber: MISSING")

try:
    import pytesseract
    print("pytesseract: OK")
except ImportError:
    print("pytesseract: MISSING")

try:
    from pdf2image import convert_from_path
    print("pdf2image: OK")
except ImportError:
    print("pdf2image: MISSING - needed for OCR")

try:
    pytesseract.get_tesseract_version()
    print("Tesseract engine: OK")
except:
    print("Tesseract engine: MISSING - install separately")