# Please enter the command in the terminal, for example: ‘accelerate launch --config_file default_config.yaml pretrain.py’ to start this code
import logging
import math
import os
import datetime
import json
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
from tqdm.auto import tqdm
import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    set_seed,
)

from accelerate import Accelerator
from accelerate.logging import get_logger

# Force loading weights without strict checking
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')


@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: str = field(
        default="/openbayes/input/input0",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use fast tokenizer"}
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={
            "help": "Override the default torch dtype",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataArguments:
    """Arguments for data processing"""
    train_file: Optional[str] = field(
        default="./data/IEA_Energy_Dataset.json",
        metadata={"help": "Path to training data file"}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to validation data file"},
    )
    block_size: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum sequence length"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "Number of processes for preprocessing"},
    )
    data_cache_dir: Optional[str] = field(
        default="./data/data_cache",
        metadata={"help": "Directory for caching processed data"},
    )


class TrainingLogger:
    """Logger for tracking training progress"""

    def __init__(self, log_dir, accelerator=None):
        self.log_dir = log_dir
        self.accelerator = accelerator
        self.is_main_process = accelerator.is_main_process if accelerator else True

        if self.is_main_process:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(log_dir, f"training_log_{get_time()}.txt")
            self.tb_writer = None
            self.setup_tensorboard()

    def setup_tensorboard(self):
        """Initialize TensorBoard writer"""
        if not self.is_main_process:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "tensorboard"))
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")

    def log_step(self, step, metrics, prefix="train"):
        """Log training step metrics"""
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
        """Log model information"""
        if not self.is_main_process:
            return

        n_params = sum(p.numel() for p in model.parameters())
        with open(self.log_file, "a") as f:
            f.write(f"Model size: {n_params / 1e6:.2f}M parameters\n")

    def close(self):
        """Close the logger"""
        if self.is_main_process and self.tb_writer:
            self.tb_writer.close()


def prepare_dataloader(train_file, validation_file, data_cache_dir, model_path,
                       use_fast_tokenizer, block_size, num_workers,
                       train_batch_size, eval_batch_size):
    """Prepare training and validation dataloaders"""

    # Load datasets
    data_files = {}
    if train_file:
        data_files["train"] = train_file
    if validation_file:
        data_files["validation"] = validation_file

    extension = train_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=data_cache_dir,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=use_fast_tokenizer,
    )

    print("Preprocessing data...")

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    # Determine sequence length
    if block_size is None:
        block_size = tokenizer.model_max_length
    else:
        block_size = min(block_size, tokenizer.model_max_length)

    def group_texts(examples):
        """Group texts into fixed-length blocks"""
        # Concatenate all texts
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Ensure total length is divisible by block_size
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        # Split by chunks of block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Process datasets
    cache_path = os.path.join(data_cache_dir or ".", f"processed_{get_time()}")
    if os.path.exists(cache_path):
        print(f"Loading processed data from cache: {cache_path}")
        processed_datasets = load_from_disk(cache_path)
    else:
        print("Processing datasets...")
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=num_workers,
            remove_columns=["text"],
        )

        processed_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=num_workers,
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"] if "validation" in processed_datasets else None

    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )

    if eval_dataset:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )
    else:
        eval_dataloader = None

    return train_dataloader, eval_dataloader, tokenizer


def prepare_model(model_path, torch_dtype):
    """Load and prepare the model"""

    # Set torch dtype
    if torch_dtype == "auto":
        dtype = None
    else:
        dtype = getattr(torch, torch_dtype) if torch_dtype else torch.bfloat16

    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
    )

    return model


def prepare_optimizer(model, learning_rate, weight_decay):
    """Create optimizer with weight decay"""

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
    )

    return optimizer


def prepare_lr_scheduler(train_dataloader, optimizer, num_epochs,
                         scheduler_type, warmup_steps, grad_accumulation_steps):
    """Create learning rate scheduler"""

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / grad_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch

    lr_scheduler = transformers.get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )
    return lr_scheduler


def main():
    # Initialize accelerator for distributed training
    accelerator = Accelerator()

    # Load configuration from file
    with open("./config.json", "r") as f:
        config = json.load(f)

    model_args = ModelArguments(
        model_name_or_path=config.get("model_name_or_path", "./model"),
        use_fast_tokenizer=config.get("use_fast_tokenizer", True),
        torch_dtype=config.get("torch_dtype", "bfloat16")
    )

    data_args = DataArguments(
        train_file=config.get("train_file", "./data/tra_data/tra_data.json"),
        validation_file=config.get("validation_file", "./data/val_data/val_data.json"),
        block_size=config.get("block_size", 1024),
        preprocessing_num_workers=config.get("preprocessing_num_workers", 4),
        data_cache_dir=config.get("data_cache_dir", "./data_cache")
    )

    # Extract training arguments from config
    training_config = {k: v for k, v in config.items()
                       if k not in [param for param in dir(model_args) if not param.startswith("_")] and
                       k not in [param for param in dir(data_args) if not param.startswith("_")]}

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

    # Print configuration
    if accelerator.is_main_process:
        logger.info("Training configuration:")
        logger.info(f"Model args: {model_args}")
        logger.info(f"Data args: {data_args}")
        logger.info(f"Training args: {training_args}")

    # Set random seed for reproducibility
    set_seed(42)

    # Initialize training logger
    training_logger = TrainingLogger(log_dir, accelerator)

    # Check for existing checkpoints
    resume_from_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        checkpoints = [folder for folder in os.listdir(training_args.output_dir)
                       if folder.startswith("checkpoint-")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            resume_from_checkpoint = os.path.join(training_args.output_dir, latest_checkpoint)
            logger.info(f"Found checkpoint, resuming from {resume_from_checkpoint}")

    gradient_accumulation_steps = training_args.gradient_accumulation_steps

    # Prepare data
    train_dataloader, eval_dataloader, tokenizer = prepare_dataloader(
        data_args.train_file,
        data_args.validation_file,
        data_args.data_cache_dir,
        model_args.model_name_or_path,
        model_args.use_fast_tokenizer,
        data_args.block_size,
        data_args.preprocessing_num_workers,
        training_args.per_device_train_batch_size,
        training_args.per_device_eval_batch_size
    )

    # Prepare model
    model = prepare_model(
        model_args.model_name_or_path,
        model_args.torch_dtype
    )

    # Resize token embeddings if needed
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Log model info
    training_logger.log_model_info(model)

    # Prepare optimizer and scheduler
    optimizer = prepare_optimizer(
        model,
        training_args.learning_rate,
        training_args.weight_decay)

    lr_scheduler = prepare_lr_scheduler(
        train_dataloader,
        optimizer,
        training_args.num_train_epochs,
        training_args.lr_scheduler_type,
        training_args.warmup_steps,
        gradient_accumulation_steps
    )

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler
    )

    # Start training
    logger.info("Starting training")
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

    # Load checkpoint if resuming
    if resume_from_checkpoint:
        accelerator.load_state(resume_from_checkpoint)

    completed_steps = 0
    starting_epoch = 0

    # Get resume info
    if resume_from_checkpoint:
        path = os.path.basename(resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]
        resume_step = int(training_difference.replace("checkpoint-", ""))
        starting_epoch = resume_step // num_update_steps_per_epoch
        resume_step -= starting_epoch * num_update_steps_per_epoch
        completed_steps = resume_step

    # Training loop
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
                        unwrapped_model.save_pretrained(
                            f"{training_args.output_dir}/model-{completed_steps}",
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                            state_dict=accelerator.get_state_dict(model),
                        )
                        # Save tokenizer on main process
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(f"{training_args.output_dir}/model-{completed_steps}")

            # Evaluate model
            if eval_dataloader is not None and accelerator.sync_gradients and completed_steps % training_args.eval_steps == 0:
                logger.info("Evaluating model")
                model.eval()
                eval_loss = 0

                for eval_step, eval_batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**eval_batch)

                    loss = outputs.loss.detach().float()
                    loss = accelerator.reduce(loss, reduction="mean")
                    eval_loss += loss

                eval_loss = eval_loss / len(eval_dataloader)
                try:
                    perplexity = math.exp(eval_loss)
                except OverflowError:
                    perplexity = float("inf")

                eval_metrics = {
                    "eval_loss": eval_loss.item(),
                    "perplexity": perplexity,
                }

                accelerator.print(f"{eval_metrics}")
                accelerator.log(eval_metrics, step=completed_steps)
                training_logger.log_step(completed_steps, eval_metrics, prefix="eval")

                model.train()

            # Stop training after specified steps
            if completed_steps >= training_args.final_steps:
                break

    # Save final model
    logger.info("Saving final model")
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        f"{training_args.output_dir}/final-model",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model)
    )

    if accelerator.is_main_process:
        tokenizer.save_pretrained(f"{training_args.output_dir}/final-model")

    # Clean up
    training_logger.close()
    logger.info(f"Training completed. Model saved to {training_args.output_dir}/final-model")


if __name__ == "__main__":
    main()