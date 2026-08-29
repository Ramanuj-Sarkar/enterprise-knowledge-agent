"""
Download SEC 10-K filings from EDGAR for a list of tickers.

SEC EDGAR is free, requires no API key, but DOES require a descriptive
User-Agent header identifying you (they'll block generic/missing ones).
Docs: https://www.sec.gov/os/accessing-edgar-data

Usage:
    pip install sec-edgar-downloader
    python download_filings.py
"""

from sec_edgar_downloader import Downloader
from pathlib import Path

# --- Configure this ---
YOUR_NAME = "Ramanuj Sarkar"
YOUR_EMAIL = "ramanuj.sarkar.ge@gmail.com"  # SEC requires a real contact in the User-Agent

# A modest starting set - S&P mix across sectors, ~20 companies is plenty
# of volume (each 10-K is 50-150+ pages) to justify distributed processing.
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "JPM", "BAC", "GS",
    "JNJ", "PFE", "UNH",
    "XOM", "CVX",
    "WMT", "PG", "KO",
    "BA", "CAT", "GE",
    "DIS", "NFLX",
]

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def main():
    dl = Downloader(YOUR_NAME, YOUR_EMAIL, str(RAW_DIR))

    for ticker in TICKERS:
        print(f"Downloading 10-K filings for {ticker}...")
        try:
            # Pull the last 3 annual filings per company - enough for
            # meaningful volume without blowing up local disk/runtime.
            dl.get("10-K", ticker, limit=3, download_details=True)
        except Exception as e:
            print(f"  Failed for {ticker}: {e}")

    print(f"\nDone. Filings saved under: {RAW_DIR}")


if __name__ == "__main__":
    main()
