#!/usr/bin/env python3
"""
SEC Case LLM Evaluation Runner

Main script to run the complete evaluation pipeline:
1. Build evaluation dataset from SEC cases (extract PDFs + ground truth)
2. Run LLM predictions on all cases
3. Calculate and display scores
"""

import argparse
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def build_dataset(args):
    """Build the evaluation dataset from SEC cases using Reducto."""
    from preprocessing.dataset_builder import build_evaluation_dataset
    
    print("=" * 60)
    print("Building Evaluation Dataset")
    print("=" * 60)
    print("Using Reducto AI for structured extraction...")
    print("Note: This uses Reducto credits (~4 credits/page)\n")
    
    result = build_evaluation_dataset(
        input_file=args.input,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    print(json.dumps(result['statistics'], indent=2))
    
    return result


def run_llm_evaluation(args):
    """Run LLM evaluation on all cases."""
    from evaluation.llm_runner import LLMRunner, MockProvider, OpenAIProvider, AnthropicProvider
    
    print("=" * 60)
    print("Running LLM Evaluation")
    print("=" * 60)
    
    # Load evaluation dataset
    dataset_file = os.path.join(args.output_dir, 'evaluation_dataset.json')
    
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file not found at {dataset_file}")
        print("Run with --build-dataset first to create the dataset.")
        return None
    
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    cases = data.get('cases', [])
    
    # Apply skip and limit
    start_idx = args.skip_cases if hasattr(args, 'skip_cases') else 0
    end_idx = start_idx + args.max_eval_cases if args.max_eval_cases else len(cases)
    cases = cases[start_idx:end_idx]
    
    print(f"Loaded {len(cases)} cases for evaluation (skipping first {start_idx})")
    
    # Select provider
    if args.provider == 'mock':
        provider = MockProvider(model_name=args.model or 'MockModel')
    elif args.provider == 'openai':
        provider = OpenAIProvider(
            model=args.model or 'gpt-4',
            api_key=args.api_key
        )
    elif args.provider == 'anthropic':
        provider = AnthropicProvider(
            model=args.model or 'claude-3-opus-20240229',
            api_key=args.api_key
        )
    else:
        print(f"Unknown provider: {args.provider}")
        return None
    
    # Run evaluation
    runner = LLMRunner(
        provider=provider,
        short_prompt=args.short_prompt,
        max_text_length=args.max_text_length
    )
    
    result = runner.run_evaluation(cases, verbose=True)
    
    # Display results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"\nModel: {result.model_name}")
    print(f"Cases Evaluated: {len(cases)}")
    print(f"Duration: {result.duration_seconds:.1f} seconds")
    print(f"\nOverall Score: {result.score.overall_score:.1f}%")
    print("\nIndividual Scores:")
    print(f"  Resolution Type: {result.score.resolution_type_accuracy:.1f}%")
    print(f"  Disgorgement:    {result.score.disgorgement_accuracy:.1f}%")
    print(f"  Penalty:         {result.score.penalty_accuracy:.1f}%")
    print(f"  Interest:        {result.score.interest_accuracy:.1f}%")
    print(f"  Monetary Avg:    {result.score.monetary_accuracy:.1f}%")
    print(f"  Injunction:      {result.score.injunction_accuracy:.1f}%")
    print(f"  Officer Bar:     {result.score.officer_bar_accuracy:.1f}%")
    print(f"  Conduct Restr:   {result.score.conduct_restriction_accuracy:.1f}%")
    
    # Save results
    if args.save_results:
        results_file = os.path.join(args.output_dir, f'evaluation_results_{args.provider}.json')
        
        # Handle append mode
        if args.append_results and os.path.exists(results_file):
            with open(results_file, 'r') as f:
                existing = json.load(f)
            
            # Append new predictions
            existing_predictions = existing.get('predictions', [])
            new_predictions = result.to_dict().get('predictions', [])
            existing_predictions.extend(new_predictions)
            existing['predictions'] = existing_predictions
            
            # Recalculate scores with merged data
            from evaluation.scoring import calculate_score
            all_cases = []
            for p in existing_predictions:
                if p.get('success'):
                    all_cases.append({
                        'predicted': p['predicted'],
                        'ground_truth': p['ground_truth']
                    })
            new_score = calculate_score(all_cases, result.model_name)
            existing['score'] = new_score.to_dict()
            existing['scorable_counts'] = existing['score'].get('scorable_counts', {})
            existing['scorable_counts']['total_cases'] = len(existing_predictions)
            
            with open(results_file, 'w') as f:
                json.dump(existing, f, indent=2, default=str)
            print(f"\nResults appended to: {results_file} (total: {len(existing_predictions)} cases)")
        else:
            with open(results_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            print(f"\nResults saved to: {results_file}")
    
    return result


def show_sample(args):
    """Show a sample case and prompt."""
    from evaluation.llm_prompt_formatter import format_prompt
    
    # Load evaluation dataset
    dataset_file = os.path.join(args.output_dir, 'evaluation_dataset.json')
    
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file not found at {dataset_file}")
        return
    
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    cases = data.get('cases', [])
    
    if not cases:
        print("No cases found in dataset.")
        return
    
    case = cases[0]
    
    print("=" * 60)
    print(f"Sample Case: {case['case_id']}")
    print("=" * 60)
    
    print(f"\nMetadata:")
    print(json.dumps(case['metadata'], indent=2))
    
    print(f"\nComplaint Text (first 1000 chars):")
    print("-" * 40)
    print(case['complaint_text'][:1000])
    print("...")
    
    print(f"\nGround Truth:")
    print(json.dumps(case['ground_truth'], indent=2))
    
    print("\n" + "=" * 60)
    print("Generated Prompt (first 2000 chars):")
    print("=" * 60)
    prompt = format_prompt(case['complaint_text'], short_format=args.short_prompt)
    print(prompt[:2000])
    if len(prompt) > 2000:
        print("...")


def main():
    parser = argparse.ArgumentParser(
        description='SEC Case LLM Evaluation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build dataset from SEC cases (uses Reducto for structured extraction)
  python run_evaluation.py --build-dataset --max-cases 8

  # Run evaluation with mock provider (for testing)
  python run_evaluation.py --evaluate --provider mock

  # Run evaluation with OpenAI GPT-4o
  python run_evaluation.py --evaluate --provider openai --model gpt-4o --save-results

  # Run evaluation with Anthropic Claude
  python run_evaluation.py --evaluate --provider anthropic --model claude-3-opus-20240229

  # Show a sample case and prompt
  python run_evaluation.py --show-sample
        """
    )
    
    # Actions
    parser.add_argument('--build-dataset', action='store_true',
                        help='Build evaluation dataset from SEC cases')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run LLM evaluation on all cases')
    parser.add_argument('--show-sample', action='store_true',
                        help='Show a sample case and prompt')
    
    # Dataset options
    parser.add_argument('--input', '-i', default='sec-cases.json',
                        help='Input SEC cases JSON file (default: sec-cases.json)')
    parser.add_argument('--output-dir', '-o', default='data/processed',
                        help='Output directory (default: data/processed)')
    parser.add_argument('--max-cases', '-m', type=int, default=None,
                        help='Maximum cases to process for dataset building')
    
    # Evaluation options
    parser.add_argument('--provider', choices=['mock', 'openai', 'anthropic'],
                        default='mock', help='LLM provider (default: mock)')
    parser.add_argument('--model', default=None,
                        help='Model name (provider-specific)')
    parser.add_argument('--api-key', default=None,
                        help='API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)')
    parser.add_argument('--max-eval-cases', type=int, default=None,
                        help='Maximum cases to evaluate (default: all)')
    parser.add_argument('--skip-cases', type=int, default=0,
                        help='Number of cases to skip (for batch continuation)')
    parser.add_argument('--append-results', action='store_true',
                        help='Append to existing results file instead of overwriting')
    parser.add_argument('--short-prompt', action='store_true',
                        help='Use shorter prompt format')
    parser.add_argument('--max-text-length', type=int, default=None,
                        help='Maximum complaint text length before truncation')
    parser.add_argument('--save-results', action='store_true',
                        help='Save evaluation results to file')
    
    args = parser.parse_args()
    
    # Run requested actions
    if args.build_dataset:
        build_dataset(args)
    
    if args.evaluate:
        run_llm_evaluation(args)
    
    if args.show_sample:
        show_sample(args)
    
    if not (args.build_dataset or args.evaluate or args.show_sample):
        parser.print_help()


if __name__ == '__main__':
    main()
