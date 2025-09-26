import os
import requests
from sec_downloader import Downloader
from sec_downloader.types import RequestedFilings
from dotenv import load_dotenv
from pathlib import Path
from config import COMPANY_NAME, EMAIL

# ensure raw folder always exists
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()
SEC_API_KEY = os.getenv('SEC_API_KEY')

# Convert HTM to PDF using direct API
def convert_htm_to_pdf(htm_url, output_filename):
    pdf_url = f"https://api.sec-api.io/filing-reader?token={SEC_API_KEY}&type=pdf&url={htm_url}"
    
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(output_filename, 'wb') as f:
            f.write(response.content)
        return True
    return False

# Download XBRL files using sec-edgar-downloader
def download_xbrl_from_metadata(metadata, save_dir=RAW_DIR):
    """
    Download XBRL files (XML, XSD, CAL, DEF, LAB, PRE) for a given filing metadata.
    """
    headers = {"User-Agent": f"{COMPANY_NAME} {EMAIL}"}
    cik = metadata.cik.lstrip("0")  # SEC path requires no leading zeros
    accession = metadata.accession_number.replace("-", "")

    base_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
    index_url = base_url + "index.json"

    resp = requests.get(index_url, headers=headers)
    resp.raise_for_status()
    items = resp.json()["directory"]["item"]

    save_path = Path(save_dir) / f"NVIDIA_{metadata.form_type}_{metadata.filing_date}"
    save_path.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for item in items:
        fname = item["name"]
        if fname.endswith((".xml", ".xsd")):
            file_url = base_url + fname
            r = requests.get(file_url, headers=headers)
            if r.status_code == 200:
                out_file = save_path / fname
                out_file.write_bytes(r.content)
                downloaded.append(out_file)
                print(f"XBRL downloaded: {fname} -> {out_file}")
            else:
                print(f"FAILED to download {fname} from {file_url} (status {r.status_code})")
    return downloaded

# Get NVIDIA filing metadata
dl = Downloader(COMPANY_NAME, EMAIL)

metadatas = dl.get_filing_metadatas(
    RequestedFilings(ticker_or_cik="NVDA", form_type="10-K", limit=2)
)

print(f"Found {len(metadatas)} filings")

# Download PDFs
print("\nDownloading PDFs...")
for metadata in metadatas:
    print(f"Processing: {metadata.form_type} from {metadata.filing_date}")
    print(metadata)
    filename = RAW_DIR / f"NVIDIA_10-K_{metadata.filing_date}.pdf"
    
    success = convert_htm_to_pdf(metadata.primary_doc_url, filename)
    
    if success:
        print(f"SUCCESS: Saved {filename}")
    else:
        print(f"FAILED: Could not convert {metadata.primary_doc_url}")

# Download XBRL files
print("\nDownloading XBRL files...")
for metadata in metadatas:
    print(f"Processing XBRL for {metadata.form_type} filed {metadata.filing_date}")
    xbrl_files = download_xbrl_from_metadata(metadata)
    print(f"Downloaded {len(xbrl_files)} XBRL files for {metadata.filing_date}")

# Check what XBRL files were created
print("\nXBRL files saved:")
for xml_file in Path("../data/raw").rglob("*.xml"):
    size = xml_file.stat().st_size / 1024
    print(f"  {xml_file.name} ({size:.1f} KB)")

print("\nDownload complete - PDFs and XBRL files ready!")

