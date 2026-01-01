"""
Dataset Builder for SEC Case LLM Evaluation

Uses Reducto AI for structured extraction from complaint PDFs,
combined with ground truth extraction for evaluation scoring.
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from .reducto_extractor import ReductoExtractor
from .ground_truth_extractor import GroundTruthExtractor


def get_complaint_url(case: Dict) -> Optional[str]:
    """Extract complaint PDF URL from case data."""
    docs = case.get('supportingDocuments', [])
    for doc in docs:
        doc_type = doc.get('type', '').lower()
        if 'complaint' in doc_type:
            return doc.get('url')
    return None


@dataclass
class ProcessedCase:
    """A fully processed case ready for LLM evaluation."""
    case_id: str
    metadata: Dict[str, Any]
    complaint_text: str  # Reducto case_synopsis - what LLM sees
    ground_truth: Dict[str, Any]  # From fullText - for comparison only
    reducto_extraction: Optional[Dict[str, Any]] = None  # Full Reducto structured data
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SkippedCase:
    """A case that was skipped due to extraction failure."""
    case_id: str
    title: str
    reason: str
    url: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetBuilder:
    """
    Builds evaluation dataset using Reducto AI for PDF extraction.
    
    Pipeline:
    1. Extract structured data from complaint PDFs via Reducto
    2. Extract ground truth outcomes from fullText
    3. Split into:
       - evaluation_dataset.json (resolved: settled + litigated)
       - prediction_dataset.json (ongoing: no resolution yet)
    """
    
    def __init__(self):
        """Initialize the dataset builder with Reducto extractor."""
        self.reducto_extractor = ReductoExtractor()
        self.ground_truth_extractor = GroundTruthExtractor()
        
    def process_case(self, case: Dict) -> Tuple[Optional[ProcessedCase], Optional[SkippedCase]]:
        """
        Process a single case using Reducto for structured extraction.
        
        Args:
            case: Raw case dictionary from sec-cases.json
            
        Returns:
            Tuple of (ProcessedCase if success, SkippedCase if failed)
        """
        case_id = case.get('releaseNumber', 'unknown')
        title = case.get('title', '')
        
        # Step 1: Get complaint URL
        complaint_url = get_complaint_url(case)
        
        if not complaint_url:
            return None, SkippedCase(
                case_id=case_id,
                title=title,
                reason='No complaint URL found in supportingDocuments',
                url=None
            )
        
        # Step 2: Extract via Reducto
        result = self.reducto_extractor.extract_from_url(complaint_url)
        
        if not result["success"]:
            return None, SkippedCase(
                case_id=case_id,
                title=title,
                reason=f'Reducto extraction failed: {result.get("error", "Unknown error")}',
                url=complaint_url
            )
        
        reducto_data = result["data"].to_dict()
        
        # Step 3: Use case_synopsis as the "complaint_text" for LLM input
        complaint_text = reducto_data.get("case_synopsis", "")
        
        if not complaint_text or len(complaint_text) < 100:
            return None, SkippedCase(
                case_id=case_id,
                title=title,
                reason='Reducto synopsis too short or empty',
                url=complaint_url
            )
        
        # Step 4: Extract ground truth from fullText (for evaluation scoring)
        full_text = case.get('features', {}).get('fullText', '')
        ground_truth = self.ground_truth_extractor.extract(full_text)
        
        # Step 5: Build processed case with Reducto data
        processed = ProcessedCase(
            case_id=case_id,
            metadata={
                'release_date': case.get('releaseDate', ''),
                'title': title,
                'complaint_url': complaint_url,
                'case_url': case.get('url', ''),
                'court': reducto_data.get('court') or case.get('features', {}).get('court', ''),
                'respondents': case.get('features', {}).get('respondents', []),
                'charges': reducto_data.get('charges') or case.get('features', {}).get('charges', []),
                'fraud_type': reducto_data.get('fraud_type'),
                'scheme_summary': reducto_data.get('scheme_summary'),
                'defendant_names': reducto_data.get('defendant_names'),
                'reducto_usage': result.get('usage', {})
            },
            complaint_text=complaint_text,
            ground_truth=ground_truth.to_dict(),
            reducto_extraction=reducto_data
        )
        
        return processed, None
    
    def build_dataset(
        self,
        input_file: str,
        output_dir: str,
        max_cases: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Build evaluation dataset with all cases.
        
        Args:
            input_file: Path to sec-cases.json
            output_dir: Directory to save output files
            max_cases: Optional limit on number of cases
            verbose: Print progress updates
            
        Returns:
            Summary statistics dict
        """
        # Load input data
        if verbose:
            print(f"Loading data from {input_file}...")
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        cases = data.get('cases', [])
        if max_cases:
            cases = cases[:max_cases]
        
        total = len(cases)
        if verbose:
            print(f"Processing {total} cases using Reducto...")
        
        # Process all cases
        processed_cases: List[ProcessedCase] = []
        skipped_cases: List[SkippedCase] = []
        total_credits = 0
        
        for i, case in enumerate(cases):
            if verbose:
                print(f"  [{i + 1}/{total}] Processing {case.get('releaseNumber', 'unknown')}...")
            
            processed, skipped = self.process_case(case)
            
            if processed:
                processed_cases.append(processed)
                # Track Reducto credits
                usage = processed.metadata.get('reducto_usage', {})
                total_credits += usage.get('credits', 0)
                if verbose:
                    print(f"    ✓ Success ({usage.get('credits', 0)} credits)")
            if skipped:
                skipped_cases.append(skipped)
                if verbose:
                    print(f"    ✗ Skipped: {skipped.reason}")
        
        # Split into resolved (for evaluation) and ongoing (for prediction)
        resolved_cases = [c for c in processed_cases 
                         if c.ground_truth.get('resolution_type') in ('settled', 'litigated')]
        ongoing_cases = [c for c in processed_cases 
                        if c.ground_truth.get('resolution_type') == 'ongoing']
        
        if verbose:
            print(f"\nExtraction complete:")
            print(f"  Successful: {len(processed_cases)}")
            print(f"    - Resolved (for evaluation): {len(resolved_cases)}")
            print(f"    - Ongoing (for prediction): {len(ongoing_cases)}")
            print(f"  Skipped: {len(skipped_cases)}")
            print(f"  Total Reducto credits used: {total_credits}")
        
        # Calculate statistics for resolved cases
        stats = self._calculate_stats(resolved_cases)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save evaluation dataset (resolved cases only - settled/litigated)
        evaluation_dataset = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'extraction_method': 'reducto',
                'total_processed': total,
                'resolved_count': len(resolved_cases),
                'ongoing_count': len(ongoing_cases),
                'skipped': len(skipped_cases),
                'total_credits_used': total_credits,
                'description': 'Resolved cases only (settled/litigated) for LLM evaluation'
            },
            'statistics': stats,
            'cases': [c.to_dict() for c in resolved_cases]
        }
        
        with open(os.path.join(output_dir, 'evaluation_dataset.json'), 'w') as f:
            json.dump(evaluation_dataset, f, indent=2)
        
        # Save prediction dataset (ongoing cases - no ground truth scoring)
        prediction_dataset = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'extraction_method': 'reducto',
                'count': len(ongoing_cases),
                'description': 'Ongoing cases for LLM prediction (no scoring - outcomes unknown)'
            },
            'cases': [c.to_dict() for c in ongoing_cases]
        }
        
        with open(os.path.join(output_dir, 'prediction_dataset.json'), 'w') as f:
            json.dump(prediction_dataset, f, indent=2)
        
        # Save skipped cases log
        skipped_data = {
            'metadata': {
                'total_skipped': len(skipped_cases),
                'created_at': datetime.now().isoformat()
            },
            'cases': [s.to_dict() for s in skipped_cases]
        }
        
        with open(os.path.join(output_dir, 'skipped_cases.json'), 'w') as f:
            json.dump(skipped_data, f, indent=2)
        
        if verbose:
            print(f"\nFiles saved to {output_dir}:")
            print(f"  - evaluation_dataset.json ({len(resolved_cases)} resolved cases)")
            print(f"  - prediction_dataset.json ({len(ongoing_cases)} ongoing cases)")
            print(f"  - skipped_cases.json ({len(skipped_cases)} cases)")
        
        return {
            'total_processed': total,
            'resolved': len(resolved_cases),
            'ongoing': len(ongoing_cases),
            'skipped': len(skipped_cases),
            'total_credits': total_credits,
            'statistics': stats
        }
    
    def _calculate_stats(self, cases: List[ProcessedCase]) -> Dict[str, Any]:
        """Calculate statistics about ground truth distribution."""
        if not cases:
            return {}
        
        resolution_counts = {}
        disgorgement_count = 0
        penalty_count = 0
        interest_count = 0
        injunction_count = 0
        officer_bar_count = 0
        conduct_restriction_count = 0
        
        for case in cases:
            gt = case.ground_truth
            
            # Resolution type counts
            res_type = gt.get('resolution_type', 'unknown')
            resolution_counts[res_type] = resolution_counts.get(res_type, 0) + 1
            
            # Monetary amount availability
            if gt.get('disgorgement_amount') is not None:
                disgorgement_count += 1
            if gt.get('penalty_amount') is not None:
                penalty_count += 1
            if gt.get('prejudgment_interest') is not None:
                interest_count += 1
            
            # Boolean flags
            if gt.get('has_injunction'):
                injunction_count += 1
            if gt.get('has_officer_director_bar'):
                officer_bar_count += 1
            if gt.get('has_conduct_restriction'):
                conduct_restriction_count += 1
        
        total = len(cases)
        
        return {
            'resolution_type_distribution': resolution_counts,
            'monetary_availability': {
                'disgorgement': {
                    'count': disgorgement_count,
                    'percentage': round(100 * disgorgement_count / total, 1)
                },
                'penalty': {
                    'count': penalty_count,
                    'percentage': round(100 * penalty_count / total, 1)
                },
                'prejudgment_interest': {
                    'count': interest_count,
                    'percentage': round(100 * interest_count / total, 1)
                }
            },
            'remedial_measures': {
                'has_injunction': {
                    'count': injunction_count,
                    'percentage': round(100 * injunction_count / total, 1)
                },
                'has_officer_director_bar': {
                    'count': officer_bar_count,
                    'percentage': round(100 * officer_bar_count / total, 1)
                },
                'has_conduct_restriction': {
                    'count': conduct_restriction_count,
                    'percentage': round(100 * conduct_restriction_count / total, 1)
                }
            }
        }


def build_evaluation_dataset(
    input_file: str,
    output_dir: str,
    max_cases: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to build the evaluation dataset using Reducto.
    
    Args:
        input_file: Path to sec-cases.json
        output_dir: Directory to save output files
        max_cases: Optional limit on number of cases
        verbose: Print progress
        
    Returns:
        Summary statistics dict
    """
    builder = DatasetBuilder()
    return builder.build_dataset(
        input_file=input_file,
        output_dir=output_dir,
        max_cases=max_cases,
        verbose=verbose
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build SEC case evaluation dataset using Reducto')
    parser.add_argument('--input', '-i', default='sec-cases.json', help='Input JSON file')
    parser.add_argument('--output', '-o', default='data/processed', help='Output directory')
    parser.add_argument('--max', '-m', type=int, default=None, help='Max cases to process')
    
    args = parser.parse_args()
    
    print("Using Reducto AI for structured extraction...")
    print("Note: This uses Reducto credits (~4 credits/page)\n")
    
    result = build_evaluation_dataset(
        input_file=args.input,
        output_dir=args.output,
        max_cases=args.max
    )
    
    print("\n" + "=" * 50)
    print("Dataset build complete!")
    print(f"Statistics: {json.dumps(result, indent=2)}")
