#!/usr/bin/env python3
"""
Import Reducto extractions and integrate with evaluation dataset.

Two modes:
1. Import from Reducto JSON export file (from web UI)
2. Download PDFs locally and process via Reducto API

Usage:
  # Import from Reducto web export
  python import_reducto.py --import-json reducto_export.json --case-id LR-26445

  # Download and process via API
  python import_reducto.py --process-cases --max-cases 10
"""

import argparse
import json
import os
import sys
import requests
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


# Headers that work for SEC PDF downloads
DOWNLOAD_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; SEC-Research-Bot/1.0)',
    'Accept': 'application/pdf,*/*',
    'Accept-Language': 'en-US,en;q=0.5',
}

PDF_CACHE_DIR = 'data/pdfs'


def download_pdf(url: str, cache_dir: str = PDF_CACHE_DIR) -> str:
    """
    Download PDF from URL and cache locally.
    
    Returns path to cached file, or None if download failed.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create filename from URL
    filename = url.split('/')[-1]
    cache_path = os.path.join(cache_dir, filename)
    
    # Check cache first
    if os.path.exists(cache_path):
        print(f"  Using cached: {cache_path}")
        return cache_path
    
    # Download
    try:
        response = requests.get(url, headers=DOWNLOAD_HEADERS, timeout=60)
        response.raise_for_status()
        
        # Verify it's a PDF
        content_type = response.headers.get('Content-Type', '')
        if response.status_code == 200:
            with open(cache_path, 'wb') as f:
                f.write(response.content)
            print(f"  Downloaded: {cache_path} ({len(response.content)} bytes)")
            return cache_path
    except Exception as e:
        print(f"  Download failed: {e}")
        return None


def process_with_reducto(pdf_path: str) -> dict:
    """
    Process a local PDF file with Reducto.
    
    Returns Reducto extraction result.
    """
    try:
        from preprocessing.reducto_extractor import ReductoExtractor
        extractor = ReductoExtractor()
        result = extractor.extract_from_file(pdf_path)
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


def import_reducto_json(json_file: str, case_id: str) -> dict:
    """
    Import a Reducto JSON export from the web UI.
    
    Args:
        json_file: Path to Reducto JSON export
        case_id: Case ID to associate with this extraction
        
    Returns:
        Reducto extraction data
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Handle both single result and array format
    if 'result' in data and isinstance(data['result'], list):
        extraction = data['result'][0] if data['result'] else {}
    else:
        extraction = data
    
    return {
        "case_id": case_id,
        "success": True,
        "data": extraction,
        "usage": data.get('usage', {}),
        "job_id": data.get('job_id', '')
    }


def update_dataset_with_reducto(case_id: str, reducto_data: dict):
    """
    Update evaluation dataset and results with Reducto extraction data.
    """
    # Load current results
    results_file = 'data/processed/evaluation_results_openai.json'
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Find the case in predictions
    updated = False
    for pred in results.get('predictions', []):
        if pred.get('case_id') == case_id:
            # Add Reducto data to metadata
            if 'metadata' not in pred:
                pred['metadata'] = {}
            
            pred['metadata']['reducto_extraction'] = reducto_data
            
            # Extract key fields for easy access
            if isinstance(reducto_data, dict):
                data = reducto_data.get('data', reducto_data)
                pred['metadata']['case_synopsis'] = data.get('case_synopsis', '')
                pred['metadata']['scheme_summary'] = data.get('scheme_summary', '')
                pred['metadata']['fraud_type'] = data.get('fraud_type', '')
                pred['metadata']['defendant_names'] = data.get('defendant_names', '')
                pred['metadata']['defendant_profit'] = data.get('defendant_profit')
                pred['metadata']['extracted_charges'] = data.get('charges', '')
            
            updated = True
            print(f"  Updated {case_id} with Reducto data")
            break
    
    if not updated:
        print(f"  Warning: Case {case_id} not found in results")
        return False
    
    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return True


def process_cases(max_cases: int = 10):
    """
    Download PDFs and process with Reducto for cases in the dataset.
    """
    # Load sec-cases.json to get PDF URLs
    with open('sec-cases.json', 'r') as f:
        sec_data = json.load(f)
    
    # Load evaluation results to get case IDs
    with open('data/processed/evaluation_results_openai.json', 'r') as f:
        results = json.load(f)
    
    case_ids = [p['case_id'] for p in results['predictions']][:max_cases]
    
    # Build URL lookup
    url_lookup = {}
    for case in sec_data['cases']:
        case_id = case.get('releaseNumber')
        for doc in case.get('supportingDocuments', []):
            if 'complaint' in doc.get('type', '').lower():
                url_lookup[case_id] = doc.get('url')
                break
    
    print(f"Processing {len(case_ids)} cases...")
    
    success_count = 0
    for case_id in case_ids:
        print(f"\n[{case_id}]")
        
        url = url_lookup.get(case_id)
        if not url:
            print(f"  No URL found")
            continue
        
        # Download PDF
        pdf_path = download_pdf(url)
        if not pdf_path:
            continue
        
        # Process with Reducto
        print(f"  Processing with Reducto...")
        result = process_with_reducto(pdf_path)
        
        if result.get('success'):
            # Update dataset
            reducto_data = result.get('data', {})
            if hasattr(reducto_data, 'to_dict'):
                reducto_data = reducto_data.to_dict()
            update_dataset_with_reducto(case_id, {"data": reducto_data})
            success_count += 1
            print(f"  ✓ Success")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
    
    print(f"\n{'='*50}")
    print(f"Processed {success_count}/{len(case_ids)} cases successfully")
    
    # Regenerate viewer
    if success_count > 0:
        print("\nRegenerating HTML viewer...")
        import generate_viewer
        generate_viewer.update_cases_html()


def main():
    parser = argparse.ArgumentParser(
        description='Import Reducto extractions into evaluation dataset'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Import JSON command
    import_parser = subparsers.add_parser('import', help='Import Reducto JSON export')
    import_parser.add_argument('json_file', help='Path to Reducto JSON export')
    import_parser.add_argument('--case-id', required=True, help='Case ID (e.g., LR-26445)')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Download and process PDFs')
    process_parser.add_argument('--max-cases', type=int, default=10, help='Max cases to process')
    
    args = parser.parse_args()
    
    if args.command == 'import':
        print(f"Importing Reducto JSON for {args.case_id}...")
        reducto_data = import_reducto_json(args.json_file, args.case_id)
        update_dataset_with_reducto(args.case_id, reducto_data)
        
        # Regenerate viewer
        import generate_viewer
        generate_viewer.update_cases_html()
        print("Done! Open cases.html to view.")
        
    elif args.command == 'process':
        process_cases(max_cases=args.max_cases)
        
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

