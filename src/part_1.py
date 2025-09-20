"""
Text extraction meeting Part 1 requirements exactly
"""
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
import json
import logging
from ocr_config import test_ocr_setup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextExtractor:
    def __init__(self):
        ocr_ok, ocr_msg = test_ocr_setup()
        if ocr_ok:
            logger.info(f"OCR ready: {ocr_msg}")
            self.ocr_available = True
        else:
            logger.warning(f"OCR not available: {ocr_msg}")
            self.ocr_available = False
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text meeting Part 1 requirements"""
        results = {
            'pdf_path': str(pdf_path),
            'total_pages': 0,
            'successful_pages': 0,
            'ocr_pages': [],
            'failed_pages': [],
            'pages': [],
            'word_boxes': []
        }
        
        with pdfplumber.open(pdf_path) as pdf:
            results['total_pages'] = len(pdf.pages)
            
            for page_num, page in enumerate(pdf.pages, 1):
                # Use layout parameters as required
                text = page.extract_text(x_density=2, y_density=2)
                
                if text and text.strip():
                    # Extract word bounding boxes as required
                    words = page.extract_words()
                    
                    results['pages'].append({
                        'page_num': page_num,
                        'text': text,
                        'method': 'pdfplumber'
                    })
                    results['successful_pages'] += 1
                    
                    # Store word bounding boxes
                    results['word_boxes'].append({
                        'page_num': page_num,
                        'words': words
                    })
                else:
                    # OCR fallback as required
                    if self.ocr_available:
                        ocr_text = self._ocr_page(pdf_path, page_num)
                        if ocr_text:
                            results['pages'].append({
                                'page_num': page_num,
                                'text': ocr_text,
                                'method': 'ocr'
                            })
                            results['ocr_pages'].append(page_num)
                        else:
                            results['failed_pages'].append(page_num)
                    else:
                        results['failed_pages'].append(page_num)
        
        return results
    
    def _ocr_page(self, pdf_path, page_num):
        """Apply OCR to specific page"""
        try:
            images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
            if images:
                text = pytesseract.image_to_string(images[0])
                return text.strip() if text else None
        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}")
        return None
    
    def save_extraction_results(self, results, output_dir="../data/parsed"):
        """Save results meeting Part 1 requirements"""
        pdf_name = Path(results['pdf_path']).stem
        output_path = Path(output_dir) / pdf_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save per-page .txt files as required
        for page_data in results['pages']:
            page_file = output_path / f"page_{page_data['page_num']:03d}.txt"
            with open(page_file, 'w', encoding='utf-8') as f:
                f.write(page_data['text'])
        
        # Save OCR log as required
        ocr_log = {
            'pdf_path': results['pdf_path'],
            'pages_requiring_ocr': results['ocr_pages'],
            'ocr_page_count': len(results['ocr_pages']),
            'total_pages': results['total_pages']
        }
        
        with open(output_path / "ocr_log.json", 'w') as f:
            json.dump(ocr_log, f, indent=2)
        
        # Save word bounding boxes as required
        with open(output_path / "word_boxes.json", 'w') as f:
            json.dump(results['word_boxes'], f, indent=2)
        
        return len(results['pages'])

def test_part1_requirements():
    """Test Part 1 against exact requirements"""
    extractor = TextExtractor()
    
    pdf_files = list(Path("../data/raw").glob("NVIDIA_*.pdf"))
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        
        results = extractor.extract_text_from_pdf(pdf_file)
        saved_count = extractor.save_extraction_results(results)
        
        print(f"Part 1 Results:")
        print(f"  Per-page .txt files: {saved_count}")
        print(f"  OCR pages logged: {len(results['ocr_pages'])}")
        print(f"  Word boxes saved: {len(results['word_boxes'])}")
        print(f"  Output directory: data/parsed/{pdf_file.stem}/")

if __name__ == "__main__":
    test_part1_requirements()