"""
Reducto-based structured extraction from SEC complaint PDFs.

Uses a pre-configured Reducto pipeline to extract:
- Case synopsis and scheme summary
- Defendant information
- Charges and fraud type
- Monetary amounts
- Relief sought
- Court and filing info
"""
import os
import tempfile
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ReductoExtraction:
    """Structured data extracted from SEC complaint."""
    case_synopsis: str
    defendant_names: str
    defendant_type: str
    is_repeat_offender: bool
    charges: str
    fraud_type: str
    scheme_summary: str
    victim_count: Optional[int]
    amount_raised: Optional[float]
    defendant_profit: Optional[float]
    violation_start_date: Optional[str]
    violation_end_date: Optional[str]
    seeks_disgorgement: bool
    seeks_penalty: bool
    seeks_injunction: bool
    seeks_officer_bar: bool
    seeks_penny_stock_bar: bool
    seeks_industry_bar: bool
    other_relief: Optional[str]
    court: str
    filing_date: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReductoExtractor:
    """Extract structured data from SEC complaints using Reducto."""
    
    # Pre-configured pipeline with SEC complaint schema
    PIPELINE_ID = "k97bn9ne73pkmqk8c0ar2pt41s7ydwy4"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize with API key from param or environment.
        
        Args:
            api_key: Reducto API key. If not provided, reads from REDUCTO_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("REDUCTO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "REDUCTO_API_KEY not found. "
                "Set it via environment variable or pass to constructor."
            )
        
        # Lazy import to avoid requiring reductoai if not used
        try:
            from reducto import Reducto
            self.client = Reducto(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "reductoai package not installed. Run: pip install reductoai"
            )
    
    # Headers to mimic a browser request (SEC blocks automated requests)
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Referer': 'https://www.sec.gov/',
    }
    
    def extract_from_url(self, pdf_url: str, timeout: int = 120) -> Dict[str, Any]:
        """
        Download PDF from URL and extract structured data.
        
        Args:
            pdf_url: URL to the PDF file
            timeout: Request timeout in seconds
            
        Returns:
            Dict with 'success', 'data' (ReductoExtraction), 'usage', 'job_id'
        """
        # Download PDF to temp file with browser headers
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                response = requests.get(
                    pdf_url, 
                    timeout=timeout, 
                    verify=True,
                    headers=self.HEADERS
                )
                response.raise_for_status()
                tmp.write(response.content)
                tmp_path = tmp.name
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Failed to download PDF: {str(e)}",
                "data": None
            }
        
        try:
            return self.extract_from_file(tmp_path)
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    
    def extract_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        Extract structured data from local PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dict with 'success', 'data' (ReductoExtraction), 'usage', 'job_id'
        """
        try:
            # Upload file to Reducto
            upload = self.client.upload(file=Path(file_path))
            
            # Run extraction pipeline
            result = self.client.pipeline.run(
                input=upload,
                pipeline_id=self.PIPELINE_ID
            )
            
            # Parse result
            if result.result and len(result.result) > 0:
                raw_data = result.result[0]
                
                # Convert to typed dataclass
                extraction = ReductoExtraction(
                    case_synopsis=raw_data.get("case_synopsis", ""),
                    defendant_names=raw_data.get("defendant_names", ""),
                    defendant_type=raw_data.get("defendant_type", "unknown"),
                    is_repeat_offender=raw_data.get("is_repeat_offender", False),
                    charges=raw_data.get("charges", ""),
                    fraud_type=raw_data.get("fraud_type", "other"),
                    scheme_summary=raw_data.get("scheme_summary", ""),
                    victim_count=raw_data.get("victim_count"),
                    amount_raised=raw_data.get("amount_raised"),
                    defendant_profit=raw_data.get("defendant_profit"),
                    violation_start_date=raw_data.get("violation_start_date"),
                    violation_end_date=raw_data.get("violation_end_date"),
                    seeks_disgorgement=raw_data.get("seeks_disgorgement", False),
                    seeks_penalty=raw_data.get("seeks_penalty", False),
                    seeks_injunction=raw_data.get("seeks_injunction", False),
                    seeks_officer_bar=raw_data.get("seeks_officer_bar", False),
                    seeks_penny_stock_bar=raw_data.get("seeks_penny_stock_bar", False),
                    seeks_industry_bar=raw_data.get("seeks_industry_bar", False),
                    other_relief=raw_data.get("other_relief"),
                    court=raw_data.get("court", ""),
                    filing_date=raw_data.get("filing_date", "")
                )
                
                return {
                    "success": True,
                    "data": extraction,
                    "raw_data": raw_data,
                    "usage": {
                        "num_pages": result.usage.num_pages if result.usage else 0,
                        "num_fields": result.usage.num_fields if result.usage else 0,
                        "credits": result.usage.credits if result.usage else 0
                    },
                    "job_id": result.job_id
                }
            else:
                return {
                    "success": False,
                    "error": "No extraction result returned",
                    "data": None,
                    "job_id": result.job_id if result else None
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Extraction failed: {str(e)}",
                "data": None
            }
    
    def extract_batch(
        self, 
        pdf_urls: list, 
        max_concurrent: int = 5,
        progress_callback=None
    ) -> list:
        """
        Extract from multiple PDFs.
        
        Args:
            pdf_urls: List of PDF URLs to process
            max_concurrent: Max concurrent extractions (not implemented yet)
            progress_callback: Optional callback(completed, total) for progress
            
        Returns:
            List of extraction results
        """
        results = []
        total = len(pdf_urls)
        
        for i, url in enumerate(pdf_urls):
            result = self.extract_from_url(url)
            result["url"] = url
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results


# CLI for testing
if __name__ == "__main__":
    import json
    import sys
    
    # Check for API key
    if not os.getenv("REDUCTO_API_KEY"):
        print("Error: REDUCTO_API_KEY environment variable not set")
        print("Run: export REDUCTO_API_KEY=your_key_here")
        sys.exit(1)
    
    # Test URL (Artur Khachatryan case)
    test_url = "https://www.sec.gov/files/litigation/complaints/2025/comp-pr2025-210.pdf"
    
    print(f"Testing Reducto extraction...")
    print(f"URL: {test_url}\n")
    
    extractor = ReductoExtractor()
    result = extractor.extract_from_url(test_url)
    
    if result["success"]:
        print("✓ Extraction successful!\n")
        print(f"Job ID: {result['job_id']}")
        print(f"Credits used: {result['usage']['credits']}")
        print(f"Pages: {result['usage']['num_pages']}")
        print(f"Fields: {result['usage']['num_fields']}\n")
        
        print("Extracted Data:")
        print("-" * 50)
        data = result["data"].to_dict()
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"{key}: {value[:100]}...")
            else:
                print(f"{key}: {value}")
    else:
        print(f"✗ Extraction failed: {result['error']}")
        sys.exit(1)

