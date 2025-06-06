# Smart Energy Large-Language-Model Evaluation System – User Guide

## System Overview
This evaluation system assesses the quality of large-language models (LLMs) in the **smart-energy** domain.  
It supports three task types:

| Task | Purpose |
|------|---------|
| **Modeling (code)** | Build smart-energy system models, run simulations, implement algorithms |
| **Explanation** | Explain smart-energy concepts, technologies, and systems |
| **QA** | Answer domain-specific questions |

Two scoring mechanisms are available (each on a 10-point scale):

| Score | Meaning |
|-------|---------|
| **A-Score** | Benchmark-based comparison against established smart-energy standards |
| **E-Score** | Independent qualitative evaluation specific to the smart-energy domain |

---

## File Structure
```text
smart_energy_evaluation.py           # Main evaluation script
config_smart_energy_modeling.json    # Modeling-task config
config_smart_energy_explanation.json # Explanation-task config
config_smart_energy_qa.json          # QA-task config
smart_energy_data.json               # Example data file
```

---

## Installation
```bash
pip install openai tqdm statistics
```

---

## Basic Usage (CLI)
```bash
# Evaluate a Modeling task with both scores
python smart_energy_evaluation.py --input smart_energy_data.json --task modeling --score_type both

# Evaluate an Explanation task with A-Score only
python smart_energy_evaluation.py --input smart_energy_data.json --task explanation --score_type a_score

# Evaluate a QA task with E-Score only
python smart_energy_evaluation.py --input smart_energy_data.json --task qa --score_type e_score
```

---

## Parameter Reference
| Flag | Description | Default |
|------|-------------|---------|
| `--input` | Path to input data file | `./smart_energy_data.json` |
| `--task` | Smart Energy Task Types | `modeling`, `explanation`, `qa`  |
| `--score_type` | Evaluation Type | `a_score`, `e_score`, `both` |
| `--results` | Detailed results file | `smart_energy_evaluation_results.json` |
| `--summary` | Aggregate summary file | `smart_energy_evaluation_summary.json` |

---

## Scoring Criteria

### Modeling / Explanation Tasks – **A-Score** (Benchmark-based)
- Smart-energy accuracy (e.g., power-flow analysis, renewable forecasting, optimisation)
- Compliance with industry standards (IEEE, IEC, etc.)
- Practical applicability
- Understanding of system constraints
- Scalability (from micro-grids to utility scale)
- Safety & reliability requirements
- Data-handling capability (formats, units, metrology)
- Integration capability (platforms & protocols)

### Modeling / Explanation Tasks – **E-Score** (Independent Quality)
- System correctness  
- Smart-energy functionality  
- Code quality & maintainability  
- Domain adaptability  
- Practical value  
- Energy-efficiency considerations  

### Explanation Task – Additional Criteria
- Technical accuracy (latest smart-energy tech)
- Domain completeness (smart grids, renewables, storage, demand response)
- Clarity for professionals
- Coverage & up-to-dateness
- Educational value

### QA Task – **A-Score** (Benchmark-based)
- Technical accuracy  
- Domain expertise  
- Completeness  
- Practical applicability  
- Industry relevance  
- Regulatory awareness  
- Integration considerations  
- Economic factors  
- Safety & reliability  
- Foresight  

### QA Task – **E-Score** (Independent Quality)
- Smart-energy accuracy  
- Practicality  
- Alignment with questions  
- Technical appropriateness  
- Clarity  
- Actionability  
- Domain relevance  

---

## Output Files
- **Detailed results** `smart_energy_evaluation_results.json` – per-sample scores  
- **Summary** `smart_energy_evaluation_summary.json` – overall statistics

---

## Precautions
1. Ensure the input data represents real smart-energy tasks.  
2. Large datasets may slow down API calls — please be patient.  
3. Scores are produced by GPT-4o and may contain subjectivity.  
4. Manually verify key evaluation results when necessary.  
5. Select the correct task type; evaluation standards differ.

---

## Troubleshooting
| Issue | Suggested Fix |
|-------|---------------|
| **API error** | Check API key and network connection |
| **File-format error** | Ensure the JSON is valid and all required fields are present |
| **Score-parsing failure** | Inspect API response; adjust prompts if needed |
| **Out-of-memory** | Split large datasets into batches |

---

*End of document*
