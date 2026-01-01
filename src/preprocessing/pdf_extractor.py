"""
PDF Text Extractor for SEC Complaint Documents

DEPRECATED: This module is no longer used in the main pipeline.
The project now uses Reducto AI for structured extraction.
See reducto_extractor.py for the current implementation.

This file is kept for reference and fallback purposes only.

---

Extracts text from complaint PDF files linked in SEC case data.
Skips cases where PDF extraction fails and logs them for transparency.
"""

import warnings
warnings.warn(
    "pdf_extractor is deprecated. Use reducto_extractor instead.",
    DeprecationWarning,
    stacklevel=2
)

import json
import os
import tempfile
import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None


@dataclass
class ExtractionResult:
    """Result of PDF text extraction attempt."""
    success: bool
    text: Optional[str]
    error: Optional[str]
    pdf_url: Optional[str]


class PDFExtractor:
    """Extracts text from SEC complaint PDFs."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check that required libraries are available."""
        if requests is None:
            raise ImportError("requests library is required. Install with: pip install requests")
        if pdfplumber is None and PyPDF2 is None:
            raise ImportError("pdfplumber or PyPDF2 is required. Install with: pip install pdfplumber")
    
    def download_pdf(self, url: str) -> Tuple[bool, Optional[bytes], Optional[str]]:
        """
        Download PDF from URL.
        
        Returns:
            Tuple of (success, pdf_bytes, error_message)
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; SEC-Research-Bot/1.0)'
            }
            response = requests.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Check if response is PDF
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' not in content_type.lower() and not url.endswith('.pdf'):
                return False, None, f"Response is not a PDF: {content_type}"
            
            return True, response.content, None
            
        except requests.exceptions.Timeout:
            return False, None, "Request timed out"
        except requests.exceptions.HTTPError as e:
            return False, None, f"HTTP error: {e.response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, None, f"Request failed: {str(e)}"
    
    def extract_text_from_bytes(self, pdf_bytes: bytes) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Extract text from PDF bytes.
        
        Returns:
            Tuple of (success, text, error_message)
        """
        # Try pdfplumber first (better extraction)
        if pdfplumber is not None:
            try:
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = tmp.name
                
                try:
                    with pdfplumber.open(tmp_path) as pdf:
                        text_parts = []
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text_parts.append(page_text)
                        
                        if text_parts:
                            return True, '\n\n'.join(text_parts), None
                        else:
                            return False, None, "No text extracted (possibly scanned image)"
                finally:
                    os.unlink(tmp_path)
                    
            except Exception as e:
                # Fall through to try PyPDF2
                pass
        
        # Try PyPDF2 as fallback
        if PyPDF2 is not None:
            try:
                import io
                pdf_file = io.BytesIO(pdf_bytes)
                reader = PyPDF2.PdfReader(pdf_file)
                
                text_parts = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                
                if text_parts:
                    return True, '\n\n'.join(text_parts), None
                else:
                    return False, None, "No text extracted (possibly scanned image)"
                    
            except Exception as e:
                return False, None, f"PDF parsing failed: {str(e)}"
        
        return False, None, "No PDF parser available"
    
    def extract_from_url(self, url: str) -> ExtractionResult:
        """
        Download and extract text from a PDF URL.
        
        Args:
            url: URL to the PDF file
            
        Returns:
            ExtractionResult with success status, text, and any error
        """
        # Download PDF
        success, pdf_bytes, error = self.download_pdf(url)
        if not success:
            return ExtractionResult(
                success=False,
                text=None,
                error=error,
                pdf_url=url
            )
        
        # Extract text
        success, text, error = self.extract_text_from_bytes(pdf_bytes)
        return ExtractionResult(
            success=success,
            text=text,
            error=error,
            pdf_url=url
        )
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted PDF text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()


def get_complaint_url(case: Dict) -> Optional[str]:
    """
    Extract complaint PDF URL from case data.
    
    Args:
        case: Case dictionary from SEC data
        
    Returns:
        URL string or None if not found
    """
    supporting_docs = case.get('supportingDocuments', [])
    
    for doc in supporting_docs:
        if doc.get('type') == 'complaint':
            return doc.get('url')
    
    return None


def process_cases(
    input_file: str,
    output_dir: str,
    max_cases: Optional[int] = None,
    verbose: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process all cases, extracting PDF text where possible.
    
    Args:
        input_file: Path to sec-cases.json
        output_dir: Directory to save output files
        max_cases: Optional limit on number of cases to process
        verbose: Whether to print progress
        
    Returns:
        Tuple of (successful_cases, skipped_cases)
    """
    # Load input data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    cases = data.get('cases', [])
    if max_cases:
        cases = cases[:max_cases]
    
    extractor = PDFExtractor()
    
    successful_cases = []
    skipped_cases = []
    
    total = len(cases)
    
    for i, case in enumerate(cases):
        case_id = case.get('releaseNumber', f'unknown_{i}')
        
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing {i + 1}/{total}...")
        
        # Get complaint URL
        complaint_url = get_complaint_url(case)
        
        if not complaint_url:
            skipped_cases.append({
                'case_id': case_id,
                'title': case.get('title', ''),
                'reason': 'No complaint URL found',
                'url': None
            })
            continue
        
        # Extract PDF text
        result = extractor.extract_from_url(complaint_url)
        
        if not result.success:
            skipped_cases.append({
                'case_id': case_id,
                'title': case.get('title', ''),
                'reason': result.error,
                'url': complaint_url
            })
            continue
        
        # Clean and store text
        cleaned_text = extractor.clean_text(result.text)
        
        if len(cleaned_text) < 100:  # Minimum text length check
            skipped_cases.append({
                'case_id': case_id,
                'title': case.get('title', ''),
                'reason': 'Extracted text too short (likely failed extraction)',
                'url': complaint_url
            })
            continue
        
        # Add to successful cases
        successful_case = {
            'case_id': case_id,
            'metadata': {
                'release_date': case.get('releaseDate', ''),
                'title': case.get('title', ''),
                'complaint_url': complaint_url,
                'url': case.get('url', '')
            },
            'complaint_text': cleaned_text,
            'full_text_for_ground_truth': case.get('features', {}).get('fullText', ''),
            'original_features': case.get('features', {})
        }
        successful_cases.append(successful_case)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'extracted_complaints.json'), 'w') as f:
        json.dump({
            'total_processed': total,
            'successful': len(successful_cases),
            'skipped': len(skipped_cases),
            'cases': successful_cases
        }, f, indent=2)
    
    with open(os.path.join(output_dir, 'skipped_cases.json'), 'w') as f:
        json.dump({
            'total_skipped': len(skipped_cases),
            'cases': skipped_cases
        }, f, indent=2)
    
    if verbose:
        print(f"\nCompleted!")
        print(f"  Successful: {len(successful_cases)}")
        print(f"  Skipped: {len(skipped_cases)}")
        print(f"  Output saved to: {output_dir}")
    
    return successful_cases, skipped_cases


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract text from SEC complaint PDFs')
    parser.add_argument('--input', '-i', default='sec-cases.json', help='Input JSON file')
    parser.add_argument('--output', '-o', default='data/processed', help='Output directory')
    parser.add_argument('--max', '-m', type=int, default=None, help='Max cases to process')
    
    args = parser.parse_args()
    
    process_cases(args.input, args.output, args.max)

