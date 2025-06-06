# General-Purpose Task Evaluation & Optimization System — User Guide

## Overview
This system can automatically **evaluate and optimise 11 categories of NLP tasks** (fact verification, reasoning, text classification, etc.). The workflow is:

1. **Evaluation** – assign 4-dimensional scores to every *instruction–output* pair  
2. **Optimisation** – perform up to **10 optimisation rounds** on data that did not pass  
3. **Classification & Storage** – categorise the results into *Pass*, *Dropped*, etc. :contentReference[oaicite:0]{index=0}

---

## Supported Task Types

| Task Code | Task Name                  | Description                    |
|-----------|---------------------------|--------------------------------|
| **FV**    | Fact Verification         | Verify factual correctness     |
| **Res**   | Reasoning                 | Logical / causal reasoning     |
| **TC**    | Text Classification       | Categorise text into labels    |
| **NER**   | Named-Entity Recognition  | Extract named entities         |
| **Sum**   | Summarisation             | Condense texts                 |
| **WS**    | Word Semantics            | Evaluate word meaning          |
| **Q&A**   | Question & Answer         | Answer questions               |
| **Exp**   | Explanation               | Provide explanatory output     |
| **ESM**   | Energy System Modelling   | Model energy systems           |
| **S-C**   | Single Choice             | Single-answer MCQ              |
| **M-C**   | Multiple Choice           | Multiple-answer MCQ            |

---
# Installation and Configuration

## 1. Install Dependencies

```bash
pip install openai tqdm
```

## 2. Configure the API Key

### Method 1 (recommended): Environment Variable

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Method 2: Modify the Code Directly

```python
# In task_evaluator.py
API_KEY = "your-actual-api-key-here"
```

## 3. Prepare the Input Data

The input JSON file should follow this format:

```json
[
  {
    "instruction": "Identify the named entities in the text: Apple Inc. was founded in Cupertino, California.",
    "input": "",
    "output": "Named entities:\n- Apple Inc. (Organization)\n- California (Location)\n- Cupertino (Location)"
  },
  {
    "instruction": "Another instruction...",
    "input": "Optional input content",
    "output": "Corresponding output content"
  }
]
```

---

# Usage

## Basic Usage

```bash
python task_evaluator.py --task NER --input input_data.json
```

## Full Parameters

```bash
python task_evaluator.py \
  --task NER \
  --input input_data.json \
  --output_dir ./results
```

## Parameter Description
- `--task` : task type (required, choose from the list above)  
- `--input` : path to the input JSON file (required)  
- `--output_dir` : output directory (optional, default `./`)  

---

## Output Files  
The system generates **three** files:

1. **Evaluation Results File** (`{TASK}_evaluation.json`) – detailed scores for every pair  
2. **Example JSON** (structure reference)  
   ```json
   [
     {
       "instruction": "Optimised instruction",
       "input"      : "Optimised input",
       "output"     : "Optimised output"
     }
   ]
   ```  
3. **Dropped Pairs** (`{TASK}_dropped_pairs.json`) – pairs that still failed after 10 optimisation attempts  

---

## Evaluation Criteria  

### Evaluation Dimensions (0 – 10 points each)
1. **Accuracy** – correctness and precision  
2. **Completeness** – thoroughness and depth  
3. **Relevance** – alignment with requirements  
4. **Practical Utility** – educational / applied value  

### Passing Criteria
- Average score ≥ 7.0 → *Pass*  
- Failed pairs undergo up to 10 optimisation attempts  
- Pairs still failing are discarded  

---

## Configuration Adjustments  

### Modify Passing Threshold
```python
PASS_THRESHOLD = 7.0  # adjust to change the pass standard
```

### Modify Maximum Optimisation Attempts
```python
MAX_OPTIMIZATION_ATTEMPTS = 10  # adjust to change optimisation rounds
```

### Modify Model Parameters
```python
MODEL_NAME  = "gpt-4o"  # target model
TEMPERATURE = 0.0       # sampling temperature
```

---

## Task-Specific Configuration  
Each task has its own evaluation standard and optimisation strategy.  
If needed, modify the corresponding entry in `TASK_CONFIGS`.

**Example – modify NER task configuration**
```python
TASK_CONFIGS["NER"]["system_prompt"] = "Your custom evaluation prompt..."

TASK_CONFIGS["NER"]["optimization_prompt"] = "Your custom optimization prompt..."
```
---

## Execution Examples  

### Process Named-Entity Recognition Data
```bash
python task_evaluator.py --task NER \
       --input ner_data.json \
       --output_dir ./ner_results
```

### Process Text Classification Data
```bash
python task_evaluator.py --task TC \
       --input classification_data.json \
       --output_dir ./tc_results
```

### Process Question-Answering Data
```bash
python task_evaluator.py --task Q&A \
       --input qa_data.json \
       --output_dir ./qa_results
```

---

## Troubleshooting  

### Common Problems
1. **API Key Error**  
   - Ensure the API key is correctly set  
   - Check environment variables or `API_KEY` in code  

2. **JSON Format Error**  
   - Ensure the input file is valid JSON  

3. **API Call Failure**  
   - Check network connection  
   - Confirm API quota is sufficient  
   - Review error logs  

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Performance Optimisation  

### Process Large Datasets
- Use batch processing  
- Adjust API call frequency  
- Consider concurrency  

### Reduce API Cost
- Use lighter models (e.g. `gpt-3.5-turbo`)  
- Optimise prompt length  

---

## Custom Evaluation Standards
Modify the evaluation function to support alternative scoring mechanisms or domain-specific criteria.

---

## Technical Support  
If you encounter issues, check:

1. Python environment and dependency versions  
2. OpenAI API permissions / limits  
3. Input data format  
4. Network connection status  

Refer to code comments and error logs for more technical details.

---

*End of document*
