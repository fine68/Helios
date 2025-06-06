import os
import json
import traceback
import pickle
import time
from typing import List, Dict, Generator, Tuple, Optional, Any

import openai
from tqdm.auto import tqdm
import re

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

INPUT_FILES: List[str] = []
OUTPUT_FILE: str = ""
CHUNK_SIZE: int = 5000
MODEL_NAME: str = ""
TEMPERATURE: float = 0.5
MAX_RETRY_ATTEMPTS: int = 3
PROGRESS_TRACKER_FILE: str = "progress_tracker.pkl"


MAX_API_RETRY_ATTEMPTS: int = 5
RETRY_DELAY_BASE: float = 2.0
RETRY_DELAY_MAX: float = 60.0
RETRY_WITH_EXPONENTIAL_BACKOFF: bool = True

# -----------------------------------------------------------------------------
# API Key (直接在此处填写)
# -----------------------------------------------------------------------------

API_KEY: str = ""  # ⚠️ 替换成你的 OpenAI API Key
openai.api_key = API_KEY

# -----------------------------------------------------------------------------
# Progress Tracking
# -----------------------------------------------------------------------------

class ProgressTracker:
    """跟踪处理进度，支持错误恢复。"""
    
    def __init__(self, tracker_file: str = PROGRESS_TRACKER_FILE):
        self.tracker_file = tracker_file
        self.doc_index: int = 0  # 当前处理的文档索引
        self.chunk_index: int = 0  # 当前处理的文档片段索引
        self.processed_docs: List[str] = []  # 已处理文档的哈希值，用于重复检测
        self.processed_count: int = 0  # 已处理片段总数
        self.insufficient_count: int = 0  # 内容不足的片段数
        self.success_count: int = 0  # 成功生成QA对的片段数
        self.retry_count: int = 0  # 重试次数统计
        
    def _update_attributes(self):
        """确保对象具有所有最新定义的属性。用于兼容性。"""
        # 检查并添加所有可能缺失的新属性
        if not hasattr(self, 'retry_count'):
            self.retry_count = 0
    
    def save(self) -> None:
        """保存当前进度到文件。"""
        try:
            # 确保对象有所有必要的属性
            self._update_attributes()
            
            with open(self.tracker_file, 'wb') as f:
                pickle.dump(self, f)
            print(f"进度已保存到 {self.tracker_file}")
        except Exception as e:
            print(f"保存进度时出错: {e}")
    
    @classmethod
    def load(cls, tracker_file: str = PROGRESS_TRACKER_FILE) -> 'ProgressTracker':
        """从文件加载进度，如不存在则创建新实例。"""
        try:
            if os.path.exists(tracker_file):
                with open(tracker_file, 'rb') as f:
                    tracker = pickle.load(f)
                
                # 兼容性处理：确保所有必要的属性都存在
                tracker._update_attributes()
                
                print(f"已从 {tracker_file} 加载进度")
                print(f"  当前位置: 文档 {tracker.doc_index + 1}, 片段 {tracker.chunk_index}")
                print(f"  已处理: {tracker.processed_count} 片段, 成功: {tracker.success_count}, 不足: {tracker.insufficient_count}, 重试: {tracker.retry_count}")
                return tracker
        except Exception as e:
            print(f"加载进度时出错: {e}")
            print("创建新的进度跟踪器")
        
        return cls(tracker_file)
    
    def update_position(self, doc_index: int, chunk_index: int) -> None:
        """更新当前处理位置。"""
        self.doc_index = doc_index
        self.chunk_index = chunk_index
        # 每更新位置就保存进度，确保能从最近位置恢复
        self.save()
    
    def record_processed(self, is_success: bool = False, is_insufficient: bool = False) -> None:
        """记录处理结果。"""
        self.processed_count += 1
        if is_success:
            self.success_count += 1
        if is_insufficient:
            self.insufficient_count += 1
        # 每处理一个片段就保存一次进度
        self.save()
    
    def record_retry(self) -> None:
        """记录重试次数。"""
        self.retry_count += 1
        self.save()
    
    def should_skip_doc(self, doc_hash: str) -> bool:
        """检查是否应跳过已处理文档。"""
        return doc_hash in self.processed_docs
    
    def mark_doc_processed(self, doc_hash: str) -> None:
        """标记文档为已处理。"""
        if doc_hash not in self.processed_docs:
            self.processed_docs.append(doc_hash)
            self.save()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def repair_json(content: str, error_info: str) -> Optional[str]:
    """尝试修复损坏的JSON内容。
    
    Args:
        content: 包含错误的JSON内容
        error_info: 错误信息，通常包含错误位置
        
    Returns:
        修复后的JSON内容，如果无法修复则返回None
    """
    try:
        # 解析错误信息以获取位置
        if "line" in error_info and "column" in error_info:
            # 尝试获取行和列信息
            line_str = error_info.split("line")[1].split("column")[0].strip()
            column_str = error_info.split("column")[1].split("(")[0].strip()
            
            try:
                line = int(line_str.rstrip(","))
                column = int(column_str)
                
                # 将内容分割成行
                lines = content.split("\n")
                
                if line <= len(lines):
                    problem_line = lines[line-1]
                    print(f"问题行 ({line}): {problem_line[:column+10]}...")
                    
                    # 常见问题1: 未终止的字符串
                    if "Unterminated string" in error_info:
                        # 查找问题位置前的最后一个引号
                        if column < len(problem_line):
                            # 在问题位置添加引号
                            fixed_line = problem_line[:column] + '"' + problem_line[column:]
                            lines[line-1] = fixed_line
                            print(f"尝试修复: 在位置 {column} 添加引号")
                            return "\n".join(lines)
                    
                    # 常见问题2: 意外的逗号或其他字符
                    if "Expecting" in error_info:
                        if column < len(problem_line):
                            # 移除可能导致问题的字符
                            fixed_line = problem_line[:column] + problem_line[column+1:]
                            lines[line-1] = fixed_line
                            print(f"尝试修复: 移除位置 {column} 的字符")
                            return "\n".join(lines)
            except ValueError:
                print(f"无法解析行列信息: {line_str}, {column_str}")
        
        # 更一般的修复方法: 如果是数组，尝试在错误位置之前找到最后一个完整的对象
        if content.strip().startswith('['):
            # 找到最后一个完整的对象位置
            last_valid_pos = content.rfind('"}')
            if last_valid_pos > 0:
                # 截断内容并添加数组结束
                truncated = content[:last_valid_pos+2]
                return truncated + "\n]"
        
        print("无法自动修复JSON内容")
        return None
    except Exception as e:
        print(f"尝试修复JSON时出错: {e}")
        return None


def load_json_with_recovery(path: str) -> Tuple[bool, Optional[list]]:
    """加载JSON文件，并尝试从解析错误中恢复。
    
    Returns:
        (成功标志, 数据(如果成功))
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        
        try:
            data = json.loads(content)
            return True, data
        except json.JSONDecodeError as e:
            error_info = str(e)
            print(f"JSON解析错误: {error_info}")
            
            # 尝试修复JSON
            for attempt in range(MAX_RETRY_ATTEMPTS):
                print(f"尝试修复JSON (尝试 {attempt+1}/{MAX_RETRY_ATTEMPTS})...")
                fixed_content = repair_json(content, error_info)
                if fixed_content:
                    try:
                        data = json.loads(fixed_content)
                        print(f"成功修复JSON!")
                        
                        # 可选: 保存修复后的JSON以备份
                        backup_path = f"{path}.fixed"
                        with open(backup_path, "w", encoding="utf-8") as f:
                            f.write(fixed_content)
                        print(f"已将修复后的JSON保存到: {backup_path}")
                        
                        return True, data
                    except json.JSONDecodeError as e2:
                        print(f"修复尝试失败: {e2}")
                        error_info = str(e2)
                
            print(f"在 {MAX_RETRY_ATTEMPTS} 次尝试后无法修复JSON，跳过此文件")
            return False, None
            
    except FileNotFoundError:
        print(f"警告: 文件 '{path}' 未找到")
        return False, None
    except Exception as e:
        print(f"加载 '{path}' 时出错: {e}")
        traceback.print_exc()
        return False, None


def load_all_documents() -> List[str]:
    """从多个输入文件加载文档。"""
    all_docs: List[str] = []
    
    for input_file in INPUT_FILES:
        try:
            print(f"正在加载 {input_file}...")
            success, data = load_json_with_recovery(input_file)
            
            if success and data:
                docs = extract_documents_from_data(data, input_file)
                all_docs.extend(docs)
                print(f"已从 {input_file} 加载 {len(docs)} 个文档")
            else:
                print(f"无法从 {input_file} 加载文档")
                
        except Exception as e:
            print(f"处理 {input_file} 时出错: {e}")
            traceback.print_exc()
    
    print(f"总共加载了 {len(all_docs)} 个文档")
    return all_docs


def extract_documents_from_data(data: list, path: str) -> List[str]:
    """从加载的数据中提取文档内容。"""
    if not isinstance(data, list):
        print(f"警告: {path} 中的数据不是列表格式，跳过")
        return []

    docs: List[str] = []
    for idx, item in enumerate(data):
        try:
            if isinstance(item, str):
                if item.strip():  # 确保不是空字符串
                    docs.append(item)
            elif isinstance(item, dict) and "text" in item:
                text = str(item["text"])
                if text.strip():  # 确保不是空字符串
                    docs.append(text)
            else:
                print(f"警告: 在 {path} 跳过第 {idx+1} 项，格式不支持")
        except Exception as e:
            print(f"处理 {path} 中第 {idx+1} 项时出错: {e}")
    
    return docs


def chunk_document(text: str, size: int = CHUNK_SIZE) -> Generator[str, None, None]:
    """将文本按指定大小切片。

    最后一个切片可能短于size；我们**绝不跨文档拼接**。
    """
    for i in range(0, len(text), size):
        yield text[i : i + size]


def calculate_retry_delay(attempt: int) -> float:
    """计算重试延迟时间，支持指数退避策略。"""
    if RETRY_WITH_EXPONENTIAL_BACKOFF:
        # 指数退避策略: base * (2^attempt)
        delay = RETRY_DELAY_BASE * (2 ** attempt)
        # 确保不超过最大延迟
        return min(delay, RETRY_DELAY_MAX)
    else:
        # 固定延迟策略
        return RETRY_DELAY_BASE


def generate_qa_with_retry(chunk: str, tracker: ProgressTracker) -> str:
    """带重试机制的QA生成函数。
    
    在遇到可重试错误时，自动尝试重新生成QA对。
    """
    retryable_errors = [
        "rate limit",
        "timeout",
        "connection",
        "socket",
        "server error",
        "service unavailable",
        "internal server error",
        "500",
        "502",
        "503",
        "504",
    ]
    
    for attempt in range(MAX_API_RETRY_ATTEMPTS):
        try:
            result = generate_qa(chunk)
            
            # 检查结果是否是API错误
            if result.startswith("ERROR:"):
                error_msg = result[6:].lower()
                
                # 检查是否是可重试的错误
                is_retryable = any(err in error_msg for err in retryable_errors)
                
                if is_retryable and attempt < MAX_API_RETRY_ATTEMPTS - 1:
                    delay = calculate_retry_delay(attempt)
                    print(f"遇到可重试错误: {error_msg}")
                    print(f"等待 {delay:.2f} 秒后重试 (尝试 {attempt+1}/{MAX_API_RETRY_ATTEMPTS})...")
                    tracker.record_retry()
                    time.sleep(delay)
                    continue
            
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # 检查是否是可重试的错误
            is_retryable = any(err in error_msg for err in retryable_errors)
            
            if is_retryable and attempt < MAX_API_RETRY_ATTEMPTS - 1:
                delay = calculate_retry_delay(attempt)
                print(f"API调用出错: {e}")
                print(f"等待 {delay:.2f} 秒后重试 (尝试 {attempt+1}/{MAX_API_RETRY_ATTEMPTS})...")
                tracker.record_retry()
                time.sleep(delay)
            else:
                # 不可重试的错误或超出最大重试次数
                print(f"API调用失败，错误: {e}")
                return f"ERROR: {str(e)}"
    
    # 超出最大重试次数
    print(f"在 {MAX_API_RETRY_ATTEMPTS} 次尝试后仍然失败")
    return f"ERROR: 超出最大重试次数 ({MAX_API_RETRY_ATTEMPTS})"


def generate_qa(chunk: str) -> str:
    """Invoke an OpenAI chat model and return the raw assistant content."""
    prompt_messages = [
    {
        "role": "system",
        "content": (

        ),
    },
    {
        "role": "user",
        "content": (


            + chunk
        ),
    },
]

    try:
        resp = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=prompt_messages,
            temperature=TEMPERATURE,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"调用API时出错: {e}")
        # 在这里我们返回错误信息而不是默认的"INSUFFICIENT_CONTENT"
        # 这样便于在上层处理中区分真正的内容不足和API错误
        return f"ERROR: {str(e)}"


def extract_qa(raw: str) -> Tuple[List[Dict[str, Any]], bool]:
    """从助手响应中解析题目、选项和答案。"""
    if raw.strip() == "INSUFFICIENT_CONTENT":
        return [], False
    if raw.startswith("ERROR:"):
        return [], True

    output: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {"question": "", "options": {}, "answer": ""}

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("Q:"):
            current["question"] = line[2:].strip()
        elif re.match(r"^[A-D]\.", line):
            letter, text = line.split(".", 1)
            current["options"][letter] = text.strip()
        elif line.startswith("Answer:"):
            current["answer"] = line.split("Answer:")[1].strip()
            if current["question"] and current["options"] and current["answer"]:
                output.append({
                    "instruction": current["question"],
                    "input": current["options"],
                    "output": current["answer"],
                })
            current = {"question": "", "options": {}, "answer": ""}

    if not output:
        print("警告: 解析响应时未找到有效的QA对")
    return output, False


def initialize_output_file(path: str = OUTPUT_FILE) -> None:
    """如果输出文件不存在，则初始化它并写入开括号。"""
    # 如果目录不存在则创建
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("[\n")
        print(f"创建新输出文件: {path}")


def save_qa(records: List[Dict[str, str]], path: str = OUTPUT_FILE) -> None:
    """将*records*作为换行分隔的JSON元素附加到*path*的列表中。"""
    if not records:
        return

    # 检查文件是否存在并有内容
    is_new_file = not os.path.exists(path)
    
    if is_new_file:
        initialize_output_file(path)

    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write(",\n")  # 后续会修复尾逗号


def finalize_output(path: str = OUTPUT_FILE) -> None:
    """移除最后的",\n"并附加闭括号。"""
    if not os.path.exists(path):
        # 如果未创建输出文件（未处理任何记录），则创建有效的空JSON数组
        with open(path, "w", encoding="utf-8") as f:
            f.write("[]")
        print(f"创建空输出文件: {path}")
        return

    # 检查文件是否有内容
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 如果文件只包含开括号，则添加闭括号
    if content.strip() == "[":
        with open(path, "w", encoding="utf-8") as f:
            f.write("[]")
        print(f"完成空输出文件: {path}")
        return
        
    # 否则，移除尾逗号并添加闭括号
    try:
        with open(path, "rb+") as f:
            f.seek(-2, os.SEEK_END)  # 移动到尾逗号",\n"之前的位置
            f.truncate()
            f.write(b"\n]")
        print(f"完成输出文件: {path}")
    except Exception as e:
        print(f"完成输出文件时出错: {e}")
        # 尝试修复文件
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if content.endswith(",\n"):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content[:-2] + "\n]")
            print(f"已修复并完成输出文件: {path}")


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def get_doc_hash(doc: str) -> str:
    """生成文档的简单哈希，用于标识已处理文档。"""
    import hashlib
    # 使用文档前100个字符和后100个字符来计算哈希
    # 这样既能区分文档，又避免对大文档计算完整哈希的开销
    content = (doc[:100] + doc[-100:]) if len(doc) > 200 else doc
    return hashlib.md5(content.encode()).hexdigest()


def process_chunk(chunk: str, tracker: ProgressTracker) -> Tuple[bool, List[Dict[str, str]]]:
    """处理单个文档片段，支持自动重试。
    
    Returns:
        (处理是否成功, 生成的QA对列表)
    """
    # 生成QA对，支持自动重试
    raw_response = generate_qa_with_retry(chunk, tracker)
    
    # 检查是否出现错误
    records, is_api_error = extract_qa(raw_response)
    
    if is_api_error:
        print(f"  API错误，处理失败")
        return False, []
    
    if raw_response == "INSUFFICIENT_CONTENT":
        tracker.record_processed(is_insufficient=True)
        return True, []
        
    if records:
        tracker.record_processed(is_success=True)
        print(f"  生成了 {len(records)} 个QA对")
        return True, records
    else:
        tracker.record_processed()
        print("  未从当前片段生成QA对")
        return True, []


def main() -> None:
    # 加载或创建进度跟踪器
    tracker = ProgressTracker.load()
    
    # 加载所有文档
    docs = load_all_documents()
    
    if not docs:
        print("没有文档可处理。创建空输出文件。")
        finalize_output()
        return
    
    has_records = False
    
    # 从上次中断的位置继续处理
    start_doc_index = tracker.doc_index
    
    for doc_index, doc in enumerate(docs[start_doc_index:], start=start_doc_index):
        print(f"\n处理文档 {doc_index+1}/{len(docs)}, 长度: {len(doc)} 字符")
        
        # 检查是否是已处理的文档
        doc_hash = get_doc_hash(doc)
        if tracker.should_skip_doc(doc_hash):
            print(f"文档 {doc_index+1} 已处理过，跳过")
            continue
        
        chunk_list = list(chunk_document(doc))
        start_chunk_index = tracker.chunk_index if doc_index == start_doc_index else 0
        
        print(f"文档共有 {len(chunk_list)} 个片段，从片段 {start_chunk_index+1} 开始处理")
        
        for chunk_index, chunk in enumerate(chunk_list[start_chunk_index:], start=start_chunk_index):
            # 更新当前处理位置
            tracker.update_position(doc_index, chunk_index)
            
            print(f"  处理片段 {chunk_index+1}/{len(chunk_list)}, 长度: {len(chunk)} 字符")
            
            # 处理片段并实现自动重试
            success, records = process_chunk(chunk, tracker)
            
            if not success:
                print(f"  处理片段失败，将在下次运行时重试此片段")
                # 保存当前进度
                tracker.save()
                continue
                
            if records:
                has_records = True
                save_qa(records)
        
        # 文档处理完成，重置片段索引，标记文档为已处理
        tracker.chunk_index = 0
        tracker.mark_doc_processed(doc_hash)
        print(f"文档 {doc_index+1} 已处理完成")
    
    # 所有文档处理完成
    finalize_output()
    
    print(f"\n处理摘要:")
    print(f"总文档数: {len(docs)}")
    print(f"处理的文档片段数: {tracker.processed_count}")
    print(f"成功生成QA对的片段数: {tracker.success_count}")
    print(f"内容不足的片段数: {tracker.insufficient_count}")
    print(f"重试次数: {tracker.retry_count}")
    
    if tracker.processed_count > 0:
        print(f"失败率: {tracker.insufficient_count/tracker.processed_count*100:.2f}% (若过高请检查输入内容)")
    
    if has_records:
        print(f"成功生成QA对并保存到 {OUTPUT_FILE}")
    else:
        print(f"未生成任何QA对。在 {OUTPUT_FILE} 创建了空文件")
    
    # 清理临时文件
    if os.path.exists(PROGRESS_TRACKER_FILE):
        os.remove(PROGRESS_TRACKER_FILE)
        print(f"已移除进度跟踪文件 {PROGRESS_TRACKER_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断，正在完成输出文件...")
        finalize_output()
        print("程序已安全终止")
    except Exception as e:
        print(f"程序执行出错: {e}")
        traceback.print_exc()
        print("尝试完成输出文件...")
        finalize_output()
        print("程序已终止")