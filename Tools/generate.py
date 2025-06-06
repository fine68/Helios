import re
import json
import argparse
from tqdm import tqdm

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class ChatMLPrompter:
    """ChatML format prompt builder and response parser for energy systems AI assistant"""

    def generate_prompt(self, instruction: str, input_text: str = None) -> str:
        system_message = (
            "<|im_start|>system\n"
            "You are an advanced AI assistant specializing in integrated energy systems. "
            "Your knowledge base includes renewable energy integration, smart grids, energy storage, "
            "low carbon energy, distributed energy system, energy policy, energy hub, and energy "
            "management system. Please provide comprehensive, accurate, and technically sound "
            "information in your responses. Keep your responses focused and concise, avoiding "
            "excessive lists of tangentially related terms."
            "<|im_end|>\n"
        )

        user_content = f"{instruction}\n{input_text}" if input_text else instruction
        prompt = f"{system_message}<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"

        return prompt

    def get_response(self, output: str) -> str:
        # Extract assistant response using regex pattern
        pattern = r"<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)"
        matches = re.findall(pattern, output, re.DOTALL)

        if matches:
            response = matches[-1].strip()
            # Clean up end token if present
            if response.endswith("<|im_end|>"):
                response = response[:-len("<|im_end|>")].strip()
        else:
            response = output

        # Handle special EIF marker - keep only content before it
        if "{EIF" in response:
            response = response.split("{EIF")[0].strip()

        return response


def setup_device_and_dtype(args):
    """Determine the appropriate device and data type for model inference"""
    if args.no_cuda or not torch.cuda.is_available() or args.device < 0:
        device = "cpu"
        dtype = torch.float32
    else:
        device = f"cuda:{args.device}"
        dtype = torch.int8 if args.do_int8 else torch.float16

    return device, dtype


def load_model_and_tokenizer(args, device, dtype):
    """Load and configure the model and tokenizer"""
    print(f'Model name: {args.model_name}')
    print(f'Model path: {args.base_model}')
    print(f'LoRA weights path: {args.lora_weights}')
    print(f'Using {device} for inference')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        load_in_8bit=args.do_int8,
        device_map="auto" if device != "cpu" and not args.do_int8 else None,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=True
    )

    if device == "cpu" or args.do_int8:
        model.to(device)

    # Load LoRA weights if specified
    if args.lora_weights:
        print(f"Loading LoRA weights from {args.lora_weights}")
        model = PeftModel.from_pretrained(model, args.lora_weights, torch_dtype=dtype)
        print("LoRA weights loaded successfully.")

    return model, tokenizer


def configure_tokenizer(tokenizer, model):
    """Configure tokenizer with special tokens and padding"""
    # Add ChatML end token if not present
    chatml_eos_token = "<|im_end|>"
    if chatml_eos_token not in tokenizer.vocab:
        tokenizer.add_tokens([chatml_eos_token], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        print(f"Added '{chatml_eos_token}' to tokenizer vocabulary.")

    # Set pad token if not available
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    chatml_eos_token_id = tokenizer.encode(chatml_eos_token, add_special_tokens=False)[0]
    return chatml_eos_token_id


def create_generation_config(args, tokenizer, chatml_eos_token_id):
    """Create generation configuration with specified parameters"""
    return GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=chatml_eos_token_id,
        do_sample=True,
        no_repeat_ngram_size=4,
    )


def process_questions(model, tokenizer, prompter, generation_config, question_data, device, args):
    """Process all questions and generate responses"""
    results = []

    for item in tqdm(question_data, desc="Processing questions"):
        instruction = item["instruction"]
        input_text = item.get("input")

        # Generate prompt
        prompt = prompter.generate_prompt(instruction, input_text)

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Generate response
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=args.max_new_tokens,
            )

        # Decode and extract response
        output_sequence = generation_output.sequences[0]
        output_text = tokenizer.decode(output_sequence, skip_special_tokens=False)
        response = prompter.get_response(output_text)

        # Store result
        results.append({
            'instruction': instruction,
            'input': input_text or "",
            'output': response
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="LLM inference script for energy systems Q&A")
    parser.add_argument("--model_name", "-n", required=True,
                        help="Name of the model, should be unique")
    parser.add_argument("--base_model", "-m", required=True,
                        help="Path to the base model")
    parser.add_argument("--lora_weights", "-l", default=None,
                        help="Path to the LoRA weights")
    parser.add_argument("--device", type=int, default=1,
                        help="GPU device ID (0, 1, etc.). Use -1 for CPU")
    parser.add_argument("--do_int8", action="store_true",
                        help="Enable 8-bit quantization")
    parser.add_argument("--low_cpu_mem_usage", action="store_true",
                        help="Enable low CPU memory usage for model loading")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Force CPU usage, disable CUDA")

    # Generation parameters
    parser.add_argument("--temperature", "-t", type=float, default=0.1)
    parser.add_argument("--top_p", "-p", type=float, default=0.75)
    parser.add_argument("--top_k", "-k", type=int, default=30)
    parser.add_argument("--num_beams", "-b", type=int, default=5)
    parser.add_argument("--max_new_tokens", "-s", type=int, default=1024)
    parser.add_argument("--repetition_penalty", type=float, default=1.8,
                        help="Penalty for token repetition")

    # I/O parameters
    parser.add_argument("--input_file", "-i", type=str, default="input_ls.json",
                        help="Input JSON file with questions")
    parser.add_argument("--output_file", "-o", type=str, default=None,
                        help="Output JSON file for results")

    args = parser.parse_args()

    # Setup device and model configuration
    device, dtype = setup_device_and_dtype(args)
    model, tokenizer = load_model_and_tokenizer(args, device, dtype)
    chatml_eos_token_id = configure_tokenizer(tokenizer, model)

    # Prepare for inference
    model.eval()
    prompter = ChatMLPrompter()
    generation_config = create_generation_config(args, tokenizer, chatml_eos_token_id)

    # Load questions
    with open(args.input_file, 'r', encoding='utf-8') as f:
        question_data = json.load(f)

    # Process all questions
    results = process_questions(model, tokenizer, prompter, generation_config,
                                question_data, device, args)

    # Save results
    output_file = args.output_file or f"qa_result_{args.model_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()