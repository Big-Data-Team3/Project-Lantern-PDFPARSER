"""
Simple SEC filing downloader - just get files downloaded
"""
import requests
import logging
from pathlib import Path
from config import COMPANY_NAME, EMAIL, DATA_DIR, validate_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SECDownloader:
    def __init__(self):
        if not validate_config():
            raise ValueError("Update .env file")
            
        self.data_dir = Path(DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.headers = {'User-Agent': f"{COMPANY_NAME} {EMAIL}"}
    
    def download_company_filings(self, ticker, filing_type="10-K", limit=2):
        """Download filings - simple approach"""
        try:
            logger.info(f"Downloading {ticker} {filing_type}")
            
            # Get company info
            response = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=self.headers
            )
            companies = response.json()
            
            # Find CIK for ticker
            cik = None
            for company in companies.values():
                if company['ticker'] == ticker.upper():
                    cik = str(company['cik_str']).zfill(10)
                    break
            
            if not cik:
                raise ValueError(f"Ticker {ticker} not found")
            
            # Get filing list
            response = requests.get(
                f'https://data.sec.gov/submissions/CIK{cik}.json',
                headers=self.headers
            )
            filings = response.json()['filings']['recent']
            
            # Find target filings
            downloaded_files = []
            count = 0
            
            for i, form in enumerate(filings['form']):
                if form == filing_type and count < limit:
                    accession = filings['accessionNumber'][i]
                    primary_doc = filings['primaryDocument'][i]
                    
                    # Download the filing
                    accession_clean = accession.replace('-', '')
                    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{primary_doc}"
                    
                    # Create output directory
                    output_dir = self.data_dir / ticker / filing_type
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Download file
                    doc_response = requests.get(doc_url, headers=self.headers)
                    
                    output_file = output_dir / f"{accession}_{primary_doc}"
                    with open(output_file, 'wb') as f:
                        f.write(doc_response.content)
                    
                    downloaded_files.append(str(output_file))
                    count += 1
                    logger.info(f"Downloaded: {primary_doc}")
            
            return {
                'success': True,
                'files': downloaded_files,
                'count': len(downloaded_files)
            }
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'files': [],
                'count': 0
            }
    
    def get_summary(self):
        """Simple file count"""
        files = list(self.data_dir.rglob("*"))
        files = [f for f in files if f.is_file()]
        
        total_size = sum(f.stat().st_size for f in files) / (1024*1024)
        
        return {
            'file_count': len(files),
            'total_size_mb': round(total_size, 1),
            'files': [str(f) for f in files]
        }

def test_downloader():
    """Simple test"""
    print("Testing SEC Downloader")
    
    downloader = SECDownloader()
    result = downloader.download_company_filings("AAPL", "10-K", 2)
    
    if result['success']:
        print(f"SUCCESS: Downloaded {result['count']} files")
        
        summary = downloader.get_summary()
        print(f"Total files: {summary['file_count']}")
        print(f"Total size: {summary['total_size_mb']} MB")
        
        return True
    else:
        print(f"FAILED: {result['error']}")
        return False

if __name__ == "__main__":
    success = test_downloader()
    print("PASSED" if success else "FAILED")