"""
Configuration management for Project LANTERN
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# SEC Downloader Settings
COMPANY_NAME = os.getenv('COMPANY_NAME', 'Northeastern University')
EMAIL = os.getenv('EMAIL', 'your-email@northeastern.edu')
DATA_DIR = "data/raw"

# Download Settings
DEFAULT_FILING_TYPES = ['10-K', '10-Q']
DEFAULT_COMPANIES = ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
MAX_FILINGS_PER_TYPE = 2

def validate_config():
    """Validate that required configuration is set"""
    issues = []
    
    if COMPANY_NAME == 'Northeastern University' and EMAIL == 'your-email@northeastern.edu':
        issues.append("Update EMAIL in .env file with your actual Northeastern email")
    
    if '@' not in EMAIL:
        issues.append("EMAIL must be a valid email address")
    
    if issues:
        print("Configuration Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    return True

def print_config():
    """Print current configuration (for debugging)"""
    print("Current Configuration:")
    print(f"  Company: {COMPANY_NAME}")
    print(f"  Email: {EMAIL}")
    print(f"  Data Directory: {DATA_DIR}")
    print(f"  Valid: {validate_config()}")

if __name__ == "__main__":
    print_config()