import os
import json
import traceback
import time
import argparse
from typing import List, Dict, Tuple, Optional, Any
import statistics

import openai
from tqdm.auto import tqdm

"""
Smart Energy Domain LLM Evaluation System
-----------------

This script evaluates instruction-output pairs for smart energy domain tasks using two scoring mechanisms:
1. A-Score: GPT-o1 benchmark-based comparative assessment against smart energy domain standards on a 10-point scale
2. E-Score: GPT-o1 independent quality assessment for smart energy domain on a 10-point scale

Tasks:
- Modeling (code): Smart energy system modeling, simulation, and algorithm implementation
- Explanation: Smart energy concepts, technologies, and system explanations
- QA: Smart energy domain question-answering pairs

Usage:
python smart_energy_evaluation.py --input your_data_file.json --task modeling --score_type both
"""

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Default paths
DEFAULT_INPUT_FILE = "./smart_energy_data.json"
RESULTS_OUTPUT_FILE = "smart_energy_evaluation_results.json"
SUMMARY_OUTPUT_FILE = "smart_energy_evaluation_summary.json"

# API configuration
MODEL_NAME = "gpt-4o"  # Using gpt-4o for evaluation
TEMPERATURE = 0.0      # Low temperature for consistent evaluation

# Retry configuration
MAX_API_RETRY_ATTEMPTS = 3
RETRY_DELAY_BASE = 2.0
RETRY_DELAY_MAX = 30.0
RETRY_WITH_EXPONENTIAL_BACKOFF = True

# Valid tasks and score types
VALID_TASKS = ["modeling", "explanation", "qa"]
VALID_SCORE_TYPES = ["a_score", "e_score", "both"]

# -----------------------------------------------------------------------------
# API Key Setup
# -----------------------------------------------------------------------------

# 直接在此处填写您的API密钥
API_KEY = "your-openai-api-key-here"  # 替换成您的 OpenAI API Key

def setup_api_key() -> None:
    """Setup API key from environment variable or hardcoded value"""
    # Priority: Use environment variable if available
    api_key = os.environ.get('OPENAI_API_KEY')
    
    # If not in environment, check if defined in script
    if not api_key:
        try:
            api_key = API_KEY
        except NameError:
            raise ValueError(
                "API key not found. Please set the OPENAI_API_KEY environment variable "
                "or define API_KEY in the script."
            )
    
    openai.api_key = api_key

# -----------------------------------------------------------------------------
# Configuration Loading
# -----------------------------------------------------------------------------

def load_task_config(task: str) -> Dict[str, str]:
    """Load task-specific prompts from config file"""
    config_file = f"config_smart_energy_{task}.json"
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file '{config_file}' not found. Using default prompts.")
        return get_default_smart_energy_config(task)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{config_file}': {e}")
        return get_default_smart_energy_config(task)

def get_default_smart_energy_config(task: str) -> Dict[str, str]:
    """Get default configuration for smart energy domain tasks"""
    default_configs = {
        "modeling": {
            "a_score_system": "You are an expert evaluator for smart energy system modeling and code generation tasks. You will compare solutions against benchmark standards in the smart energy domain and provide a comparative assessment on a 10-point scale.",
            "a_score_user": "Evaluate the following smart energy modeling code solution using benchmark-based comparative assessment. Consider smart energy domain best practices, industry standards, and established benchmarks in energy system modeling, smart grid technologies, renewable energy integration, and energy management systems.",
            "e_score_system": "You are an expert evaluator for smart energy system modeling and code generation tasks. You will independently assess the quality of smart energy solutions on a 10-point scale.",
            "e_score_user": "Evaluate the following smart energy modeling code solution independently for overall quality in the smart energy domain. Consider correctness, efficiency, domain-specific best practices, and practical applicability in real smart energy systems."
        },
        "explanation": {
            "a_score_system": "You are an expert evaluator for smart energy domain explanations. You will compare explanations against benchmark standards in smart energy education and technical documentation and provide a comparative assessment on a 10-point scale.",
            "a_score_user": "Evaluate the following smart energy explanation using benchmark-based comparative assessment. Compare against established standards in smart energy education, technical documentation, and industry best practices for explaining smart grid, renewable energy, energy storage, and energy management concepts.",
            "e_score_system": "You are an expert evaluator for smart energy domain explanations. You will independently assess the quality of smart energy explanations on a 10-point scale.",
            "e_score_user": "Evaluate the following smart energy explanation independently for overall quality. Consider accuracy of energy concepts, clarity for the target audience, completeness of technical details, and practical relevance to smart energy applications."
        },
        "qa": {
            "a_score_system": "You are an expert evaluator for smart energy domain question-answering pairs. You will compare answers against benchmark standards in smart energy technical support and documentation and provide a comparative assessment on a 10-point scale.",
            "a_score_user": "Evaluate the following smart energy Q&A pair using benchmark-based comparative assessment. Compare against established benchmarks for smart energy technical support, industry documentation, and expert knowledge in smart grids, renewable energy systems, energy efficiency, and energy management.",
            "e_score_system": "You are an expert evaluator for smart energy domain question-answering pairs. You will independently assess the quality of smart energy answers on a 10-point scale.",
            "e_score_user": "Evaluate the following smart energy Q&A pair independently for overall quality. Consider technical accuracy, completeness for smart energy applications, practical relevance, and usefulness for smart energy professionals or students."
        }
    }
    return default_configs.get(task, default_configs["qa"])

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def load_data_pairs(input_file: str) -> List[Dict[str, str]]:
    """Load instruction-output pairs from input JSON file."""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list, got {type(data).__name__}")
        
        valid_pairs = []
        for idx, item in enumerate(data):
            try:
                if not isinstance(item, dict):
                    print(f"Warning: Item {idx+1} is not a dictionary, skipping")
                    continue
                
                if "instruction" not in item or "output" not in item:
                    print(f"Warning: Item {idx+1} missing required fields, skipping")
                    continue
                
                if "input" not in item:
                    item["input"] = ""
                
                valid_pairs.append(item)
            except Exception as e:
                print(f"Error processing item {idx+1}: {e}")
        
        print(f"Loaded {len(valid_pairs)} valid smart energy domain pairs from {input_file}")
        return valid_pairs
    
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{input_file}': {e}")
        return []
    except Exception as e:
        print(f"Error loading '{input_file}': {e}")
        traceback.print_exc()
        return []

def calculate_retry_delay(attempt: int) -> float:
    """Calculate delay between retry attempts, with exponential backoff option."""
    if RETRY_WITH_EXPONENTIAL_BACKOFF:
        delay = RETRY_DELAY_BASE * (2 ** attempt)
        return min(delay, RETRY_DELAY_MAX)
    else:
        return RETRY_DELAY_BASE

def evaluate_with_score(data_pair: Dict[str, str], score_type: str, config: Dict[str, str]) -> Optional[float]:
    """
    Evaluate a single smart energy domain data pair using specified scoring mechanism.
    
    Args:
        data_pair: The instruction-output pair to evaluate
        score_type: Either 'a_score' or 'e_score'
        config: Configuration containing prompts for the task
    
    Returns:
        Score as float (1-10) or None if evaluation failed
    """
    system_prompt = config[f"{score_type}_system"]
    user_template = config[f"{score_type}_user"]
    
    # Construct the full prompt with smart energy context
    user_prompt = (
        f"{user_template}\n\n"
        f"SMART ENERGY TASK:\n{data_pair['instruction']}\n\n"
        f"CONTEXT/INPUT:\n{data_pair.get('input', 'N/A')}\n\n"
        f"SOLUTION/OUTPUT:\n{data_pair['output']}\n\n"
        f"Please evaluate this smart energy domain content and provide only a numerical score from 1 to 10 "
        f"(can include decimals like 7.5). Consider domain-specific requirements and industry standards."
    )
    
    # Handle API retries
    for attempt in range(MAX_API_RETRY_ATTEMPTS):
        try:
            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            resp = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=prompt_messages,
                temperature=TEMPERATURE,
            )
            
            result = resp.choices[0].message.content.strip()
            
            # Extract numerical score
            try:
                # Try to extract number from response
                import re
                numbers = re.findall(r'\d+\.?\d*', result)
                if numbers:
                    score = float(numbers[0])
                    if 1 <= score <= 10:
                        return score
                    else:
                        print(f"Warning: Score {score} out of range [1,10], attempting to clip")
                        return max(1, min(10, score))
                else:
                    print(f"Warning: No numerical score found in response: {result}")
                    return None
            except ValueError:
                print(f"Warning: Could not parse score from response: {result}")
                return None
            
        except Exception as e:
            error_msg = str(e).lower()
            retryable_errors = ["rate limit", "timeout", "connection", "server error"]
            is_retryable = any(err in error_msg for err in retryable_errors)
            
            if is_retryable and attempt < MAX_API_RETRY_ATTEMPTS - 1:
                delay = calculate_retry_delay(attempt)
                print(f"API call error: {e}")
                print(f"Waiting {delay:.2f} seconds before retry (attempt {attempt+1}/{MAX_API_RETRY_ATTEMPTS})...")
                time.sleep(delay)
            else:
                print(f"API call failed: {str(e)}")
                return None
    
    return None

def save_results(results: List[Dict], summary: Dict, results_file: str, summary_file: str):
    """Save evaluation results and summary to files."""
    try:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Smart Energy Domain LLM Evaluation System")
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE, help="Input JSON file path")
    parser.add_argument("--task", choices=VALID_TASKS, required=True, help="Smart energy task type to evaluate")
    parser.add_argument("--score_type", choices=VALID_SCORE_TYPES, default="both", help="Scoring mechanism")
    parser.add_argument("--results", default=RESULTS_OUTPUT_FILE, help="Results output file")
    parser.add_argument("--summary", default=SUMMARY_OUTPUT_FILE, help="Summary output file")
    args = parser.parse_args()
    
    # Setup API key
    try:
        setup_api_key()
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Load task configuration
    config = load_task_config(args.task)
    
    # Load data pairs
    data_pairs = load_data_pairs(args.input)
    if not data_pairs:
        print("No valid smart energy domain data pairs found. Exiting.")
        return
    
    # Determine which scores to compute
    compute_a_score = args.score_type in ["a_score", "both"]
    compute_e_score = args.score_type in ["e_score", "both"]
    
    # Process data pairs
    results = []
    a_scores = []
    e_scores = []
    
    print(f"Evaluating {len(data_pairs)} smart energy domain pairs for task '{args.task}' with score type '{args.score_type}'...")
    
    for idx, data_pair in enumerate(tqdm(data_pairs, desc=f"Evaluating Smart Energy {args.task.title()} Task")):
        result = {
            "id": idx + 1,
            "instruction": data_pair["instruction"],
            "input": data_pair.get("input", ""),
            "output": data_pair["output"]
        }
        
        # Compute A-Score if requested
        if compute_a_score:
            a_score = evaluate_with_score(data_pair, "a_score", config)
            result["a_score"] = a_score
            if a_score is not None:
                a_scores.append(a_score)
        
        # Compute E-Score if requested
        if compute_e_score:
            e_score = evaluate_with_score(data_pair, "e_score", config)
            result["e_score"] = e_score
            if e_score is not None:
                e_scores.append(e_score)
        
        results.append(result)
        
        # Print progress
        if (idx + 1) % 10 == 0 or idx == len(data_pairs) - 1:
            print(f"Progress: {idx + 1}/{len(data_pairs)} pairs evaluated.")
    
    # Calculate summary statistics
    summary = {
        "domain": "smart_energy",
        "task": args.task,
        "score_type": args.score_type,
        "total_pairs": len(data_pairs),
        "successful_evaluations": {}
    }
    
    if compute_a_score and a_scores:
        summary["a_score_stats"] = {
            "count": len(a_scores),
            "mean": round(statistics.mean(a_scores), 2),
            "median": round(statistics.median(a_scores), 2),
            "std_dev": round(statistics.stdev(a_scores) if len(a_scores) > 1 else 0, 2),
            "min": min(a_scores),
            "max": max(a_scores)
        }
        summary["successful_evaluations"]["a_score"] = len(a_scores)
    
    if compute_e_score and e_scores:
        summary["e_score_stats"] = {
            "count": len(e_scores),
            "mean": round(statistics.mean(e_scores), 2),
            "median": round(statistics.median(e_scores), 2),
            "std_dev": round(statistics.stdev(e_scores) if len(e_scores) > 1 else 0, 2),
            "min": min(e_scores),
            "max": max(e_scores)
        }
        summary["successful_evaluations"]["e_score"] = len(e_scores)
    
    # Save results
    save_results(results, summary, args.results, args.summary)
    
    # Print summary
    print("\nSmart Energy Domain Evaluation Complete!")
    print(f"Task: {args.task}")
    print(f"Total pairs processed: {len(data_pairs)}")
    
    if compute_a_score and a_scores:
        print(f"A-Score - Successfully evaluated: {len(a_scores)}/{len(data_pairs)}")
        print(f"A-Score - Average: {statistics.mean(a_scores):.2f}")
    
    if compute_e_score and e_scores:
        print(f"E-Score - Successfully evaluated: {len(e_scores)}/{len(data_pairs)}")
        print(f"E-Score - Average: {statistics.mean(e_scores):.2f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()