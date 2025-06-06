# Please enter the command in the terminal, for example: ‘accelerate launch --config_file default_config.yaml finetuning.py’ to start this code
import logging
import math
import os
import sys
import datetime
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from tqdm.auto import tqdm
import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    set_seed,
    get_scheduler,
)

# Import PEFT components for LoRA
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

# Import Accelerate components
from accelerate import Accelerator
from accelerate.logging import get_logger

# Force PyTorch to load weights without validation
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


def get_timestamp():
    """Generate timestamp string for file naming."""
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default="/openbayes/input/input1",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use fast tokenizer"}
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": "Model torch dtype",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8-bit precision"}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load model in 4-bit precision"}
    )
    enable_torch_compile: bool = field(
        default=False,
        metadata={"help": "Enable torch.compile for model acceleration"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    train_file: Optional[str] = field(
        default="./data/instruction_train.json",
        metadata={"help": "Path to training data file"}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to validation data file"},
    )
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum sequence length"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "Number of processes for preprocessing"},
    )
    data_cache_dir: Optional[str] = field(
        default="./data_cache",
        metadata={"help": "Directory for data cache"},
    )


@dataclass
class LoraArguments:
    """LoRA configuration arguments."""
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA scaling parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"}
    )
    target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "List of module names to apply LoRA to"}
    )
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for LoRA. Can be 'none', 'all' or 'lora_only'"}
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint"}
    )


class TrainingLogger:
    """Logger for training progress and metrics."""

    def __init__(self, log_dir, accelerator=None):
        """
        Initialize training logger.

        Args:
            log_dir (str): Directory for log files
            accelerator (Accelerator, optional): Accelerator instance
        """
        self.log_dir = log_dir
        self.accelerator = accelerator
        self.is_main_process = accelerator.is_main_process if accelerator else True

        if self.is_main_process:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, f"training_log_{get_timestamp()}.txt")
            self.tb_writer = None
            self._setup_tensorboard()

    def _setup_tensorboard(self):
        """Setup TensorBoard logger."""
        if not self.is_main_process:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "tensorboard"))
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")

    def log_step(self, step, metrics, prefix="train"):
        """
        Log training step metrics.

        Args:
            step (int): Current training step
            metrics (dict): Training metrics
            prefix (str): Log prefix, default is 'train'
        """
        if not self.is_main_process:
            return

        # Write to text log
        with open(self.log_file, "a") as f:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            f.write(f"[{timestamp}] {prefix} step {step}: {metrics_str}\n")

        # Write to TensorBoard
        if self.tb_writer:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(f"{prefix}/{name}", value, step)

    def log_model_info(self, model):
        """
        Log model information.

        Args:
            model (torch.nn.Module): Training model
        """
        if not self.is_main_process:
            return

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        with open(self.log_file, "a") as f:
            f.write(f"Model size: {total_params / 1e6:.2f}M parameters\n")
            f.write(f"Trainable parameters: {trainable_params / 1e6:.2f}M\n")
            f.write(f"Trainable parameters percentage: {100 * trainable_params / total_params:.2f}%\n")

    def close(self):
        """Close logger."""
        if self.is_main_process and self.tb_writer:
            self.tb_writer.close()


def prepare_instruction_dataset(tokenizer, data_args, per_device_train_batch_size, per_device_eval_batch_size):
    """
    Prepare data loaders for instruction fine-tuning using chat template format.
    """
    # Load dataset
    data_files = {}
    if data_args.train_file:
        data_files["train"] = data_args.train_file
    if data_args.validation_file:
        data_files["validation"] = data_args.validation_file

    extension = data_args.train_file.split(".")[-1]
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=data_args.data_cache_dir,
    )

    # Determine sequence length
    if tokenizer.model_max_length and tokenizer.model_max_length < 1e20:
        max_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    else:
        max_length = data_args.max_seq_length
    print(f"Using max_length: {max_length}")

    print("Preprocessing data (using chat_template format)...")

    def preprocess_instruction_data(examples):
        conversations = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples.get("input", [""] * len(examples["instruction"]))[i]
            output = examples["output"][i]

            # Build message format for chat_template
            messages = [
                {"role": "system",
                 "content": "You are Helios, created by ZJU_SET.You are a helpful AI assistant. Please follow the user's instructions carefully and provide accurate responses."}
            ]

            # Build user message
            if input_text.strip():
                user_content = f"{instruction}\n\n{input_text}"
            else:
                user_content = instruction

            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": output})

            # Format conversation using chat_template
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            conversations.append(conversation)

        # Tokenize conversations
        tokenized_conversations = tokenizer(
            conversations,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids_tensor = tokenized_conversations["input_ids"]
        labels_tensor = input_ids_tensor.clone()

        # Handle PAD token labels
        if tokenizer.pad_token_id is not None:
            labels_tensor[input_ids_tensor == tokenizer.pad_token_id] = -100
        else:
            print("Warning: tokenizer.pad_token_id is undefined, PAD token labels may not be properly ignored.")

        # Mask system and user messages for each sample
        for i in range(len(conversations)):
            current_instruction = examples["instruction"][i]
            current_input_text = examples.get("input", [""] * len(examples["instruction"]))[i]

            # Build messages to mask (system + user + assistant start token)
            messages_to_mask = [
                {"role": "system",
                 "content": "You are Helios, created by ZJU_SET.You are a helpful AI assistant. Please follow the user's instructions carefully and provide accurate responses."}
            ]

            user_content = f"{current_instruction}\n\n{current_input_text}" if current_input_text.strip() else current_instruction
            messages_to_mask.append({"role": "user", "content": user_content})

            # Generate prompt for masking with assistant start token
            prompt_for_masking = tokenizer.apply_chat_template(
                messages_to_mask,
                tokenize=False,
                add_generation_prompt=True  # This adds <|im_start|>assistant\n
            )

            # Tokenize prompt part to calculate length
            tokenized_prompt = tokenizer(
                prompt_for_masking,
                add_special_tokens=True,
                max_length=max_length,
                truncation=True,
                padding=False
            )
            prompt_length = len(tokenized_prompt["input_ids"])

            # Find actual content start position
            content_start_idx = 0
            if tokenizer.pad_token_id is not None:
                non_pad_mask = (input_ids_tensor[i] != tokenizer.pad_token_id)
                if torch.any(non_pad_mask):
                    content_start_idx = non_pad_mask.nonzero(as_tuple=True)[0][0].item()
                else:
                    continue

            # Calculate mask end position
            mask_end_idx = content_start_idx + prompt_length
            if content_start_idx < mask_end_idx <= labels_tensor.shape[1]:
                labels_tensor[i, content_start_idx:mask_end_idx] = -100
            elif mask_end_idx <= content_start_idx and prompt_length > 0:
                print(
                    f"Warning: Sample {i} prompt masking calculation may have issues. content_start_idx={content_start_idx}, prompt_length={prompt_length}")

        tokenized_conversations["labels"] = labels_tensor
        return tokenized_conversations

    # Process dataset
    cache_path = os.path.join(data_args.data_cache_dir or ".", f"processed_instruction_chat_template_{get_timestamp()}")
    if os.path.exists(cache_path):
        print(f"Loading processed data from cache: {cache_path}")
        processed_datasets = load_from_disk(cache_path)
    else:
        print("Processing dataset...")
        processed_datasets = raw_datasets.map(
            preprocess_instruction_data,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
        )
        # Save processed dataset to cache
        processed_datasets.save_to_disk(cache_path)

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets.get("validation")

    # Create data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=per_device_train_batch_size, shuffle=True, collate_fn=default_data_collator
    )
    eval_dataloader = None
    if eval_dataset:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=per_device_eval_batch_size, shuffle=False, collate_fn=default_data_collator
        )

    return train_dataloader, eval_dataloader


def setup_lora_model(base_model, model_args, lora_args):
    """
    Prepare model with LoRA configuration.
    """
    model = base_model

    # Prepare model for k-bit training if specified
    if model_args.load_in_8bit or model_args.load_in_4bit:
        print("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        print("Model prepared for k-bit training.")

    # Configure LoRA
    print(f"Configuring LoRA with modules_to_save: {lora_args.modules_to_save}")
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_args.target_modules,
        modules_to_save=lora_args.modules_to_save
    )

    # Apply LoRA to model
    print("Applying LoRA to model...")
    peft_model = get_peft_model(model, lora_config)
    print("LoRA applied.")
    peft_model.print_trainable_parameters()

    return peft_model


def create_optimizer(model, learning_rate, weight_decay):
    """Create optimizer for LoRA fine-tuning."""
    # For LoRA, we only need to optimize trainable parameters
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and "bias" not in n],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and "bias" in n],
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)


def create_lr_scheduler(train_dataloader, optimizer, num_train_epochs, lr_scheduler_type, warmup_steps,
                        gradient_accumulation_steps):
    """Create learning rate scheduler."""
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    return get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )


def main():
    # Initialize Accelerator before using any accelerate functionality
    accelerator = Accelerator()

    # Parse configuration from JSON file
    with open("./config.json", "r") as f:
        config = json.load(f)

    model_args = ModelArguments(
        model_name_or_path=config.get("model_name_or_path", "./model"),
        use_fast_tokenizer=config.get("use_fast_tokenizer", True),
        torch_dtype=config.get("torch_dtype", "bfloat16"),
        load_in_8bit=config.get("load_in_8bit", False),
        load_in_4bit=config.get("load_in_4bit", False),
        enable_torch_compile=config.get("enable_torch_compile", False)
    )

    data_args = DataArguments(
        train_file=config.get("train_file", "./data/instruction_train.json"),
        validation_file=config.get("validation_file", "./data/instruction_val.json"),
        max_seq_length=config.get("max_seq_length", 1024),
        preprocessing_num_workers=config.get("preprocessing_num_workers", 4),
        data_cache_dir=config.get("data_cache_dir", "./data_cache")
    )

    lora_args = LoraArguments(
        lora_r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("target_modules",
                                  ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        bias=config.get("bias", "none"),
        modules_to_save=config.get("modules_to_save", None)
    )

    # Extract training arguments from config
    training_config = {k: v for k, v in config.items()
                       if k not in [param for param in dir(model_args) if not param.startswith("_")] and
                       k not in [param for param in dir(data_args) if not param.startswith("_")] and
                       k not in [param for param in dir(lora_args) if not param.startswith("_")]}

    training_args = type("TrainingArgs", (), training_config)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger = get_logger(__name__)

    # Create log directory
    log_dir = os.path.join(training_args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Print current configuration
    if accelerator.is_main_process:
        logger.info("Current training configuration:")
        logger.info(f"Model arguments: {model_args}")
        logger.info(f"Data arguments: {data_args}")
        logger.info(f"LoRA arguments: {lora_args}")
        logger.info(f"Training arguments: {training_args}")

    # Set random seed
    set_seed(42)

    # Create training logger
    training_logger = TrainingLogger(log_dir, accelerator)

    # Check for latest checkpoint
    resume_from_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        checkpoints = [folder for folder in os.listdir(training_args.output_dir)
                       if folder.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_from_checkpoint = os.path.join(training_args.output_dir, latest_checkpoint)
            logger.info(f"Found checkpoint, resuming training from {resume_from_checkpoint}")

    gradient_accumulation_steps = training_args.gradient_accumulation_steps

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",
    )

    # Set eos_token to <|im_end|>
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    logger.info(f"Set eos_token to: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    # Add special tokens if not already added
    chatml_tokens = ["<|im_start|>", "<|im_end|>"]
    num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": chatml_tokens})
    logger.info(f"Added {num_added_tokens} new special tokens to tokenizer.")

    # Set pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"tokenizer.pad_token was None, set to tokenizer.eos_token: {tokenizer.eos_token}")
        else:
            # If eos_token is also None, add a default pad token
            default_pad_token = "[PAD]"
            logger.warning(
                f"tokenizer.pad_token and tokenizer.eos_token are None. Adding default pad token '{default_pad_token}'.")
            tokenizer.add_special_tokens({'pad_token': default_pad_token})
            if default_pad_token not in chatml_tokens:
                num_added_tokens += 1

    # Load base model
    torch_dtype_for_load = None
    if model_args.torch_dtype == "auto":
        torch_dtype_for_load = "auto"
    elif model_args.torch_dtype:
        try:
            torch_dtype_for_load = getattr(torch, model_args.torch_dtype)
        except AttributeError:
            logger.warning(f"Invalid torch_dtype '{model_args.torch_dtype}' specified. Using None.")
            torch_dtype_for_load = None

    logger.info(
        f"Loading base model: {model_args.model_name_or_path} with torch_dtype: {model_args.torch_dtype} -> {torch_dtype_for_load}")

    model_kwargs = {"torch_dtype": torch_dtype_for_load}
    if model_args.load_in_8bit or model_args.load_in_4bit:
        model_kwargs["load_in_8bit"] = model_args.load_in_8bit
        model_kwargs["load_in_4bit"] = model_args.load_in_4bit

    base_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    logger.info("Base model loaded.")

    # Resize model token embeddings if necessary
    original_embedding_size = base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > original_embedding_size:
        logger.info(
            f"Resizing model token embeddings from {original_embedding_size} to {len(tokenizer)} because tokenizer vocab size is larger.")
        base_model.resize_token_embeddings(len(tokenizer))
    elif num_added_tokens > 0:
        logger.info(
            f"{num_added_tokens} new tokens were added. Resizing embeddings to ensure consistency (current size: {len(tokenizer)}).")
        base_model.resize_token_embeddings(len(tokenizer))

    # Setup LoRA model
    model = setup_lora_model(base_model, model_args, lora_args)

    # Prepare instruction dataset
    train_dataloader, eval_dataloader = prepare_instruction_dataset(
        tokenizer,
        data_args,
        training_args.per_device_train_batch_size,
        training_args.per_device_eval_batch_size
    )

    # Apply torch.compile for acceleration if enabled
    if model_args.enable_torch_compile and torch.__version__ >= "2" and sys.platform != "win32":
        logger.info("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
            logger.info("Model compilation successful!")
        except Exception as e:
            logger.warning(f"Model compilation failed, continuing with uncompiled model: {e}")

    # Log model information
    training_logger.log_model_info(model)

    # Create optimizer
    optimizer = create_optimizer(model, training_args.learning_rate, training_args.weight_decay)

    # Create learning rate scheduler
    lr_scheduler = create_lr_scheduler(
        train_dataloader,
        optimizer,
        training_args.num_train_epochs,
        training_args.lr_scheduler_type,
        training_args.warmup_steps,
        gradient_accumulation_steps
    )

    # Prepare model, optimizer and data loaders with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Start training
    logger.info("Starting LoRA instruction fine-tuning")
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

    # Load checkpoint if available
    if resume_from_checkpoint:
        accelerator.load_state(resume_from_checkpoint)

    completed_steps = 0
    starting_epoch = 0

    # Get resume information
    if resume_from_checkpoint:
        path = os.path.basename(resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]
        resume_step = int(training_difference.replace("checkpoint-", ""))
        starting_epoch = resume_step // num_update_steps_per_epoch
        resume_step -= starting_epoch * num_update_steps_per_epoch
        completed_steps = resume_step

    for epoch in range(starting_epoch, int(training_args.num_train_epochs)):
        model.train()

        for step, batch in enumerate(train_dataloader):
            # Skip completed steps when resuming
            if resume_from_checkpoint and epoch == starting_epoch and step < resume_step * gradient_accumulation_steps:
                continue

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                output = model(**batch)
                loss = output.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()

                if accelerator.sync_gradients:
                    completed_steps += 1

                    # Log training metrics
                    if completed_steps % training_args.logging_steps == 0:
                        accelerator.reduce(loss, "mean")
                        metrics = {
                            "loss": loss.item(),
                            "lr": lr_scheduler.get_last_lr()[0],
                            "epoch": epoch + (step + 1) / len(train_dataloader),
                            "step": completed_steps,
                        }
                        accelerator.print(f"{metrics}")
                        accelerator.log(metrics, step=completed_steps)
                        training_logger.log_step(completed_steps, metrics)

                    # Save checkpoint periodically
                    if completed_steps % training_args.save_steps == 0:
                        logger.info(f"Saving checkpoint to {training_args.output_dir}/checkpoint-{completed_steps}")
                        accelerator.save_state(f"{training_args.output_dir}/checkpoint-{completed_steps}")
                        unwrapped_model = accelerator.unwrap_model(model)
                        # Save LoRA adapter
                        unwrapped_model.save_pretrained(
                            f"{training_args.output_dir}/adapter-{completed_steps}",
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                        )
                        # Save tokenizer on main process
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(f"{training_args.output_dir}/adapter-{completed_steps}")

            # Evaluate model
            if eval_dataloader is not None and accelerator.sync_gradients and completed_steps % training_args.eval_steps == 0:
                logger.info("Evaluating model")
                model.eval()
                eval_loss = 0
                total_correct = 0
                total_tokens = 0

                for eval_step, eval_batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**eval_batch)

                    # Calculate loss
                    loss = outputs.loss.detach().float()
                    loss = accelerator.reduce(loss, reduction="mean")
                    eval_loss += loss

                    # Calculate accuracy
                    logits = outputs.logits  # [batch_size, seq_len, vocab_size]
                    labels = eval_batch["labels"]  # [batch_size, seq_len]

                    # Exclude positions marked with -100 from accuracy calculation
                    active_mask = labels != -100  # [batch_size, seq_len]
                    pred_token_ids = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]

                    # Calculate number of correct predictions
                    correct_preds = (pred_token_ids == labels) & active_mask
                    total_correct += correct_preds.sum().item()
                    total_tokens += active_mask.sum().item()

                # Calculate average loss and accuracy
                eval_loss = eval_loss / len(eval_dataloader)
                accuracy = total_correct / total_tokens if total_tokens > 0 else 0

                try:
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                eval_metrics = {
                    "eval_loss": eval_loss.item(),
                    "perplexity": perplexity,
                    "accuracy": accuracy
                }

                accelerator.print(f"{eval_metrics}")
                accelerator.log(eval_metrics, step=completed_steps)
                training_logger.log_step(completed_steps, eval_metrics, prefix="eval")

                model.train()

            if completed_steps >= max_train_steps:
                break

    logger.info("Save final model")
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)

    # Save LoRA adapter
    unwrapped_model.save_pretrained(
        f"{training_args.output_dir}/final-adapter",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )

    if accelerator.is_main_process:
        tokenizer.save_pretrained(f"{training_args.output_dir}/final-adapter")

    # Optional: Save merged model (if needed)
    if training_args.save_merged_model:
        logger.info("Save merged model")
        # Merge LoRA weights to base model
        try:
            merged_model = unwrapped_model.merge_and_unload()
            # Save merged model
            merged_model.save_pretrained(
                f"{training_args.output_dir}/merged_model",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
            )
            logger.info(f"Merged model saved successfully: {training_args.output_dir}/merged_model")
        except Exception as e:
            logger.error(f"Failed to save merged model: {e}")

    # Close logger
    training_logger.close()
    logger.info(f"Training completed. LoRA adapter saved to {training_args.output_dir}/final-adapter")


if __name__ == "__main__":
    main()