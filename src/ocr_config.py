"""
Cross-platform OCR configuration
"""
import pytesseract
import platform
import shutil
from pathlib import Path

def configure_tesseract():
    """Auto-configure Tesseract for different platforms"""
    
    # First, try to find tesseract in PATH
    tesseract_path = shutil.which('tesseract')
    if tesseract_path:
        return True  # Already in PATH, no configuration needed
    
    # Platform-specific fallback paths
    system = platform.system()
    
    if system == "Windows":
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            Path.home() / "AppData/Local/Programs/Tesseract-OCR/tesseract.exe"
        ]
    elif system == "Darwin":  # macOS
        possible_paths = [
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
            "/usr/bin/tesseract"
        ]
    else:  # Linux
        possible_paths = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract"
        ]
    
    # Try each path
    for path in possible_paths:
        if Path(path).exists():
            pytesseract.pytesseract.tesseract_cmd = str(path)
            return True
    
    return False

def test_ocr_setup():
    """Test if OCR is properly configured"""
    try:
        version = pytesseract.get_tesseract_version()
        return True, f"Tesseract {version} found"
    except Exception as e:
        return False, f"OCR setup failed: {e}"

# Auto-configure on import
if not configure_tesseract():
    print("WARNING: Tesseract not found. OCR fallback will not work.")