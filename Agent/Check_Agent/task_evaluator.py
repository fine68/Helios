import os
import json
import traceback
import time
import argparse
from typing import List, Dict, Tuple, Optional, Any
import openai
from tqdm.auto import tqdm

"""
通用任务质量评估与优化系统
-----------------

支持的任务类型:
- FactVerification (FV)
- Reasoning (Res)  
- TextClassification (TC)
- NamedEntityRecognition (NER)
- Summarization (Sum)
- WordSemantics (WS)
- QuestionAndAnswers (Q&A)
- Explanation (Exp)
- EnergySystemModeling (ESM)
- Single-Choice (S-C)
- Multiple-Choice (M-C)

使用方法:
python task_evaluator.py --task NER --input input.json
"""

# -----------------------------------------------------------------------------
# 配置部分
# -----------------------------------------------------------------------------

# API 配置
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.0
MAX_API_RETRY_ATTEMPTS = 3
RETRY_DELAY_BASE = 2.0
RETRY_DELAY_MAX = 30.0

# 优化配置
MAX_OPTIMIZATION_ATTEMPTS = 10
PASS_THRESHOLD = 7.0  # 平均分数阈值

# API Key - 请替换为您的密钥
API_KEY = "your-api-key-here"

# -----------------------------------------------------------------------------
# 任务配置字典
# -----------------------------------------------------------------------------

TASK_CONFIGS = {
    "FV": {
        "name": "Fact Verification",
        "system_prompt": """You are evaluating Fact Verification instruction-output pairs for educational datasets. 
These pairs should demonstrate accurate fact-checking abilities, proper evidence evaluation, and clear reasoning about truth claims.

EVALUATION CRITERIA (0-10 scale):
1. ACCURACY: Correctness of fact verification and evidence assessment
2. COMPLETENESS: Thoroughness in checking claims and providing evidence  
3. RELEVANCE: Appropriateness of verification methods and sources cited
4. PRACTICAL_UTILITY: Educational value for learning fact-checking skills

A pair passes if the average score ≥ 7.0.""",
        
        "optimization_prompt": """Improve this Fact Verification pair by:
1. Enhancing accuracy of fact-checking process
2. Adding more comprehensive evidence evaluation
3. Improving clarity of verification reasoning
4. Increasing educational value for fact-checking learning"""
    },
    
    "Res": {
        "name": "Reasoning", 
        "system_prompt": """You are evaluating Reasoning instruction-output pairs for educational datasets.
These pairs should demonstrate logical thinking, step-by-step problem solving, and clear argumentation.

EVALUATION CRITERIA (0-10 scale):
1. ACCURACY: Correctness of logical reasoning and conclusions
2. COMPLETENESS: Thoroughness of reasoning steps and consideration of alternatives
3. RELEVANCE: Appropriateness of reasoning approach to the problem type
4. PRACTICAL_UTILITY: Educational value for learning reasoning skills

A pair passes if the average score ≥ 7.0.""",
        
        "optimization_prompt": """Improve this Reasoning pair by:
1. Strengthening logical flow and step-by-step thinking
2. Adding more comprehensive analysis of the problem
3. Improving clarity of reasoning process
4. Increasing educational value for reasoning skill development"""
    },
    
    "TC": {
        "name": "Text Classification",
        "system_prompt": """You are evaluating Text Classification instruction-output pairs for educational datasets.
These pairs should demonstrate accurate categorization, proper feature identification, and clear classification reasoning.

EVALUATION CRITERIA (0-10 scale):
1. ACCURACY: Correctness of classification and category assignments
2. COMPLETENESS: Thoroughness in feature analysis and classification rationale
3. RELEVANCE: Appropriateness of classification approach and categories used
4. PRACTICAL_UTILITY: Educational value for learning text classification methods

A pair passes if the average score ≥ 7.0.""",
        
        "optimization_prompt": """Improve this Text Classification pair by:
1. Enhancing accuracy of classification decisions
2. Adding more detailed feature analysis and reasoning
3. Improving explanation of classification methodology
4. Increasing educational value for classification learning"""
    },
    
    "NER": {
        "name": "Named Entity Recognition",
        "system_prompt": """You are evaluating Named Entity Recognition instruction-output pairs for educational datasets.
These pairs should demonstrate accurate entity identification, proper classification, and comprehensive entity detection.

EVALUATION CRITERIA (0-10 scale):
1. ACCURACY: Correctness of entity identification and classification
2. COMPLETENESS: Thoroughness in finding all relevant entities
3. RELEVANCE: Appropriateness of entity types and classification scheme  
4. PRACTICAL_UTILITY: Educational value for learning NER concepts

A pair passes if the average score ≥ 7.0.""",
        
        "optimization_prompt": """Improve this Named Entity Recognition pair by:
1. Enhancing accuracy of entity identification and classification
2. Adding more comprehensive entity detection
3. Improving explanation of NER methodology
4. Increasing educational value for NER learning"""
    },
    
    "Sum": {
        "name": "Summarization",
        "system_prompt": """You are evaluating Summarization instruction-output pairs for educational datasets.
These pairs should demonstrate accurate content condensation, key point identification, and coherent summary structure.

EVALUATION CRITERIA (0-10 scale):
1. ACCURACY: Correctness and faithfulness to original content
2. COMPLETENESS: Coverage of important points and balanced representation
3. RELEVANCE: Appropriateness of content selection and summary focus
4. PRACTICAL_UTILITY: Educational value for learning summarization skills

A pair passes if the average score ≥ 7.0.""",
        
        "optimization_prompt": """Improve this Summarization pair by:
1. Enhancing accuracy and faithfulness to source content
2. Adding more comprehensive coverage of key points
3. Improving summary structure and coherence
4. Increasing educational value for summarization learning"""
    },
    
    "WS": {
        "name": "Word Semantics",
        "system_prompt": """You are evaluating Word Semantics instruction-output pairs for educational datasets.
These pairs should demonstrate accurate semantic analysis, proper word relationship understanding, and clear meaning explanations.

EVALUATION CRITERIA (0-10 scale):
1. ACCURACY: Correctness of semantic analysis and word meaning interpretation
2. COMPLETENESS: Thoroughness in exploring semantic relationships and contexts
3. RELEVANCE: Appropriateness of semantic analysis approach
4. PRACTICAL_UTILITY: Educational value for learning semantic concepts

A pair passes if the average score ≥ 7.0.""",
        
        "optimization_prompt": """Improve this Word Semantics pair by:
1. Enhancing accuracy of semantic analysis
2. Adding more comprehensive exploration of meaning relationships
3. Improving clarity of semantic explanations
4. Increasing educational value for semantics learning"""
    },
    
    "Q&A": {
        "name": "Question and Answers",
        "system_prompt": """You are evaluating Question-Answer instruction-output pairs for educational datasets.
These pairs should demonstrate accurate information retrieval, comprehensive answering, and clear explanations.

EVALUATION CRITERIA (0-10 scale):
1. ACCURACY: Correctness of answers and information provided
2. COMPLETENESS: Thoroughness in addressing all aspects of questions
3. RELEVANCE: Appropriateness of answers to questions asked
4. PRACTICAL_UTILITY: Educational value for learning and knowledge acquisition

A pair passes if the average score ≥ 7.0.""",
        
        "optimization_prompt": """Improve this Question-Answer pair by:
1. Enhancing accuracy and correctness of answers
2. Adding more comprehensive coverage of question aspects
3. Improving clarity and organization of responses
4. Increasing educational and informational value"""
    },
    
    "Exp": {
        "name": "Explanation",
        "system_prompt": """You are evaluating Explanation instruction-output pairs for educational datasets.
These pairs should demonstrate clear concept explanation, logical structure, and effective teaching methods.

EVALUATION CRITERIA (0-10 scale):
1. ACCURACY: Correctness of explanations and concept presentation
2. COMPLETENESS: Thoroughness in covering explanation requirements
3. RELEVANCE: Appropriateness of explanation approach and examples used
4. PRACTICAL_UTILITY: Educational effectiveness and clarity for learners

A pair passes if the average score ≥ 7.0.""",
        
        "optimization_prompt": """Improve this Explanation pair by:
1. Enhancing accuracy and clarity of explanations
2. Adding more comprehensive concept coverage
3. Improving logical structure and flow
4. Increasing educational effectiveness and accessibility"""
    },
    
    "ESM": {
        "name": "Energy System Modeling",
        "system_prompt": """You are evaluating Energy System Modeling instruction-output pairs for educational datasets.
These pairs should demonstrate accurate modeling concepts, proper system analysis, and practical applications.

EVALUATION CRITERIA (0-10 scale):
1. ACCURACY: Correctness of modeling approaches and technical content
2. COMPLETENESS: Thoroughness in system analysis and modeling steps
3. RELEVANCE: Appropriateness of modeling methods for energy systems
4. PRACTICAL_UTILITY: Educational value for learning energy system concepts

A pair passes if the average score ≥ 7.0.""",
        
        "optimization_prompt": """Improve this Energy System Modeling pair by:
1. Enhancing technical accuracy of modeling content
2. Adding more comprehensive system analysis
3. Improving practical applicability and examples
4. Increasing educational value for energy system learning"""
    },
    
    "S-C": {
        "name": "Single Choice",
        "system_prompt": """You are evaluating Single-Choice question instruction-output pairs for educational datasets.
These pairs should demonstrate clear question formulation, accurate answer selection, and proper explanation.

EVALUATION CRITERIA (0-10 scale):
1. ACCURACY: Correctness of answer choice and reasoning
2. COMPLETENESS: Thoroughness in explaining choice rationale
3. RELEVANCE: Appropriateness of question difficulty and options
4. PRACTICAL_UTILITY: Educational value for assessment and learning

A pair passes if the average score ≥ 7.0.""",
        
        "optimization_prompt": """Improve this Single-Choice pair by:
1. Enhancing accuracy of correct answer and explanations
2. Adding more comprehensive reasoning for answer choice
3. Improving question clarity and option quality
4. Increasing educational and assessment value"""
    },
    
    "M-C": {
        "name": "Multiple Choice",
        "system_prompt": """You are evaluating Multiple-Choice question instruction-output pairs for educational datasets.
These pairs should demonstrate clear question formulation, accurate answer selection, and comprehensive explanations.

EVALUATION CRITERIA (0-10 scale):
1. ACCURACY: Correctness of answer choices and reasoning
2. COMPLETENESS: Thoroughness in explaining all relevant choices
3. RELEVANCE: Appropriateness of question complexity and options
4. PRACTICAL_UTILITY: Educational value for assessment and learning

A pair passes if the average score ≥ 7.0.""",
        
        "optimization_prompt": """Improve this Multiple-Choice pair by:
1. Enhancing accuracy of answer selections and explanations
2. Adding more comprehensive analysis of all options
3. Improving question clarity and choice quality
4. Increasing educational and assessment effectiveness"""
    }
}

# -----------------------------------------------------------------------------
# 核心功能函数
# -----------------------------------------------------------------------------

def setup_api_key():
    """设置API密钥"""
    api_key = os.environ.get('OPENAI_API_KEY') or API_KEY
    if not api_key or api_key == "your-api-key-here":
        raise ValueError("请设置OPENAI_API_KEY环境变量或在代码中提供API密钥")
    openai.api_key = api_key

def load_data(input_file: str) -> List[Dict]:
    """加载输入数据"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"期望JSON列表，得到 {type(data).__name__}")
            
        valid_pairs = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"警告: 项目 {idx+1} 不是字典，跳过")
                continue
            
            if "instruction" not in item or "output" not in item:
                print(f"警告: 项目 {idx+1} 缺少必需字段，跳过")
                continue
                
            if "input" not in item:
                item["input"] = ""
                
            valid_pairs.append(item)
        
        print(f"从 {input_file} 加载了 {len(valid_pairs)} 个有效数据对")
        return valid_pairs
        
    except Exception as e:
        print(f"加载文件错误: {e}")
        return []

def call_api_with_retry(messages: List[Dict], max_retries: int = MAX_API_RETRY_ATTEMPTS) -> str:
    """带重试的API调用"""
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            if attempt < max_retries - 1:
                delay = RETRY_DELAY_BASE * (2 ** attempt)
                delay = min(delay, RETRY_DELAY_MAX)
                print(f"API调用失败，{delay}秒后重试: {e}")
                time.sleep(delay)
            else:
                raise e

def evaluate_pair(pair: Dict, task_config: Dict) -> Dict:
    """评估单个数据对"""
    system_prompt = task_config["system_prompt"]
    
    user_prompt = f"""
请评估以下指令-输出对:

指令: {pair['instruction']}
输入: {pair.get('input', '')}
输出: {pair['output']}

请按以下JSON格式返回评估结果:
{{
    "scores": {{
        "accuracy": 浮点数(0-10),
        "completeness": 浮点数(0-10), 
        "relevance": 浮点数(0-10),
        "practical_utility": 浮点数(0-10),
        "average": 浮点数(0-10)
    }},
    "evaluation": "详细评估说明",
    "passed": 布尔值,
    "optimization_notes": "改进建议(如果未通过)"
}}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        result = call_api_with_retry(messages)
        # 解析JSON结果
        eval_data = json.loads(result)
        
        # 确保结果包含必要字段
        result_pair = pair.copy()
        result_pair.update(eval_data)
        
        return result_pair
        
    except Exception as e:
        print(f"评估失败: {e}")
        # 返回失败的默认结果
        result_pair = pair.copy()
        result_pair.update({
            "scores": {"accuracy": 0, "completeness": 0, "relevance": 0, "practical_utility": 0, "average": 0},
            "evaluation": f"评估失败: {str(e)}",
            "passed": False,
            "optimization_notes": "评估过程中出现错误"
        })
        return result_pair

def optimize_pair(pair: Dict, task_config: Dict, attempt: int) -> Optional[Dict]:
    """优化数据对"""
    system_prompt = f"""你是一个专业的教育数据集优化专家。
{task_config['optimization_prompt']}

请基于评估反馈优化指令-输出对，使其达到高质量标准。
"""

    user_prompt = f"""
当前数据对:
指令: {pair['instruction']}
输入: {pair.get('input', '')}
输出: {pair['output']}

评估反馈: {pair.get('evaluation', '')}
改进建议: {pair.get('optimization_notes', '')}

这是第 {attempt} 次优化尝试。请提供优化后的版本:

请按以下JSON格式返回:
{{
    "instruction": "优化后的指令",
    "input": "优化后的输入(如果有)",
    "output": "优化后的输出"
}}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        result = call_api_with_retry(messages)
        optimized_data = json.loads(result)
        
        # 创建优化后的数据对
        optimized_pair = {
            "instruction": optimized_data["instruction"],
            "input": optimized_data.get("input", ""),
            "output": optimized_data["output"],
            "optimization_attempt": attempt,
            "original_pair": pair
        }
        
        return optimized_pair
        
    except Exception as e:
        print(f"优化失败: {e}")
        return None

def save_results(data: List[Dict], filename: str):
    """保存结果到文件"""
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description="通用任务质量评估与优化系统")
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIGS.keys()), 
                       help="任务类型")
    parser.add_argument("--input", required=True, help="输入JSON文件路径")
    parser.add_argument("--output_dir", default="./output", help="输出目录")
    
    args = parser.parse_args()
    
    # 设置API密钥
    try:
        setup_api_key()
    except ValueError as e:
        print(f"错误: {e}")
        return
    
    # 获取任务配置
    task_config = TASK_CONFIGS[args.task]
    print(f"开始处理任务: {task_config['name']}")
    
    # 加载数据
    data_pairs = load_data(args.input)
    if not data_pairs:
        print("没有找到有效数据，退出")
        return
    
    # 初始化结果列表
    evaluation_results = []  # 文件A: 所有评估结果
    passed_pairs = []        # 文件B: 通过的数据对
    dropped_pairs = []       # 文件C: 被丢弃的数据对
    
    print(f"开始评估 {len(data_pairs)} 个数据对...")
    
    for idx, pair in enumerate(tqdm(data_pairs, desc="处理数据对")):
        current_pair = pair.copy()
        
        # 首次评估
        evaluated_pair = evaluate_pair(current_pair, task_config)
        evaluation_results.append(evaluated_pair)
        
        if evaluated_pair["passed"]:
            # 直接通过
            passed_pair = {
                "instruction": evaluated_pair["instruction"],
                "input": evaluated_pair.get("input", ""),
                "output": evaluated_pair["output"]
            }
            passed_pairs.append(passed_pair)
        else:
            # 需要优化
            optimized = False
            for attempt in range(1, MAX_OPTIMIZATION_ATTEMPTS + 1):
                print(f"  优化数据对 {idx+1}, 尝试 {attempt}/{MAX_OPTIMIZATION_ATTEMPTS}")
                
                # 尝试优化
                optimized_pair = optimize_pair(evaluated_pair, task_config, attempt)
                if not optimized_pair:
                    print(f"    优化失败")
                    continue
                
                # 重新评估优化后的数据对
                re_evaluated = evaluate_pair(optimized_pair, task_config)
                
                if re_evaluated["passed"]:
                    # 优化成功
                    passed_pair = {
                        "instruction": re_evaluated["instruction"],
                        "input": re_evaluated.get("input", ""),
                        "output": re_evaluated["output"]
                    }
                    passed_pairs.append(passed_pair)
                    optimized = True
                    print(f"    优化成功!")
                    break
                else:
                    # 继续下次优化
                    evaluated_pair = re_evaluated
                    print(f"    优化后评分: {re_evaluated['scores']['average']:.2f}")
            
            if not optimized:
                # 优化失败，丢弃
                dropped_pairs.append(evaluated_pair)
                print(f"  数据对 {idx+1} 优化失败，已丢弃")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluation_file = os.path.join(args.output_dir, f"{args.task}_evaluation_results.json")
    passed_file = os.path.join(args.output_dir, f"{args.task}_passed_pairs.json") 
    dropped_file = os.path.join(args.output_dir, f"{args.task}_dropped_pairs.json")
    
    save_results(evaluation_results, evaluation_file)
    save_results(passed_pairs, passed_file)
    save_results(dropped_pairs, dropped_file)
    
    # 打印总结
    print(f"\n处理完成!")
    print(f"总数据对: {len(data_pairs)}")
    print(f"通过: {len(passed_pairs)} ({len(passed_pairs)/len(data_pairs)*100:.2f}%)")
    print(f"丢弃: {len(dropped_pairs)} ({len(dropped_pairs)/len(data_pairs)*100:.2f}%)")
    print(f"\n结果文件:")
    print(f"  评估结果: {evaluation_file}")
    print(f"  通过的数据对: {passed_file}")
    print(f"  丢弃的数据对: {dropped_file}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"意外错误: {e}")
        traceback.print_exc()