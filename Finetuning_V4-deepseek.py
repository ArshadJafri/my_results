"""
Centaur-PTSD LoRA finetuning + logging script (optional). Using the combined files
from information from the Stroop task and Fear Conditional task - only few subjects added until now..

This script:
- Parses hyperparameters from the command line
  (learning rate, max_seq_length, batch sizes, grad accumulation, weight decay, epochs).
- Downloads private JSONL train/test splits from Hugging Face
  (cognition PTSD task datasets) and enforces a fixed schema.
- Loads the `marcelbinz/Llama-3.1-Centaur-70B-adapter` base model with transformers,
  enabling 4-bit loading and LoRA adapters on attention/MLP blocks.
- Trains the model with `Trainer` using a completion-only LM collator
  (with <<response>> style templates) and saves checkpoints to a scratch directory
  named by the current hyperparameters.
- Optionally reloads the finetuned model into a `transformers.pipeline`
  for quick text-generation tests.
- Searches for `trainer_state.json` in the output/checkpoint folders and,
  if found, parses `log_history` to plot and save training loss vs. steps.
"""

import os
os.environ['DS_BUILD_OPS'] = '0'
os.environ['DS_SKIP_CUDA_CHECK'] = '1'


from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
import torch
import sys
from dataclasses import dataclass, field
from typing import Optional
from transformers import set_seed
from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Features, Value
import json
import matplotlib.pyplot as plt
import argparse
from huggingface_hub import hf_hub_download, notebook_login

# --- REPOSITORY AND FILE CONSTANTS ---
TRAIN_REPO_ID = "arshad101/cognition_task_psych"
TRAIN_FILE_NAME = "combined_train_fixed.jsonl"

# -- set here the preffixed path for saving the model checkpoints
preffix_folder = "/scratch/axs7716Arsh/"

# --- ARGUMENT PARSING ---
# Filter non-flag arguments to correctly parse command-line hyperparameters
args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]

# Check for the correct number of arguments before accessing them
if len(args) < 7:
    print("Error: Missing command-line arguments.")
    print("Usage: python Finetuning_V3.py <learning_rate> <max_seq_length> <train_batch_size> <eval_batch_size> <gradient_accumulator> <weight_decay> <num_train_epochs>")
    sys.exit(1)

# -- get the parameter here
try:
    learning_rate = float(args[0])
    max_seq_length = int(args[1])
    train_batch_size = int(args[2])
    eval_batch_size = int(args[3])
    gradient_accumulation_steps = int(args[4])
    weight_decay = float(args[5])
    num_train_epochs = int(args[6])
except ValueError:
    print("Error: Command-line arguments must be valid numbers.")
    sys.exit(1)

# Define the mandatory schema to force consistent data types
custom_features = Features({
    'text': Value('string'),
    'experiment': Value('string'),
    'participant': Value('string'),
})

# --- DATASET LOADING (Initial: needed for the script to load before `main`) ---

# 1. Download the private file to the local cache using authentication
try:
    train_file_path = hf_hub_download(
        repo_id=TRAIN_REPO_ID,
        filename=TRAIN_FILE_NAME,
        repo_type="dataset",
        token=True
    )

    # 2. Load dataset from the local path, using the JSON builder and the fixed schema.
    raw_train_dataset = load_dataset(
        'json',
        data_files={'train': train_file_path},
        features=custom_features,
    )

    # This is loaded here but will be overwritten in `main` after tokenizing
    train_dataset = raw_train_dataset['train'].shuffle() 
except Exception as e:
    # If this fails, the script will likely crash in `main`, but we continue for structural completeness.
    print(f"Initial dataset check failed (will re-attempt in main): {e}")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"})
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0)


@dataclass
class DataTrainingArguments:
    dataset_text_field: str = field(default="text")
    max_seq_length: Optional[int] = field(default=max_seq_length)


def main(model_args, data_args, training_args):
    set_seed(training_args.seed)

    # Define the mandatory schema to force consistent data types (repeated for clarity/safety in main)
    custom_features_main = Features({
        'text': Value('string'),
        'experiment': Value('string'),
        'participant': Value('string'),
    })

    # --- REPOSITORY AND FILE CONSTANTS ---
    TRAIN_REPO_ID = "arshad101/cognition_task_psych"
    TRAIN_FILE_NAME = "combined_train_fixed.jsonl"
    EVAL_REPO_ID = "arshad101/cognition_task_psych_test"
    EVAL_FILE_NAME = "combined_test_fixed.jsonl"

    # 1. Download the private files to the local cache using authentication
    train_file_path = hf_hub_download(
        repo_id=TRAIN_REPO_ID,
        filename=TRAIN_FILE_NAME,
        repo_type="dataset",
        token=True
    )
    eval_file_path = hf_hub_download(
        repo_id=EVAL_REPO_ID,
        filename=EVAL_FILE_NAME,
        repo_type="dataset",
        token=True
    )

    # 2. Load datasets from local paths
    raw_train_dataset = load_dataset(
        'json',
        data_files={'train': train_file_path},
        features=custom_features_main,
    )
    train_dataset = raw_train_dataset['train'].shuffle()

    raw_eval_dataset = load_dataset(
        'json',
        data_files={'test': eval_file_path},
        features=custom_features_main,
    )
    eval_dataset = raw_eval_dataset['test'].shuffle()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples[data_args.dataset_text_field],
            truncation=True,
            max_length=data_args.max_seq_length,
            padding=False,
        )

    print("Tokenizing train dataset...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "experiment", "participant"]
    )

    print("Tokenizing eval dataset...")
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "experiment", "participant"]
    )

    # Load model directly with transformers 
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare for k-bit training
    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    
    # Get the token IDs for the response template
    # Assuming the target pattern is " [PROMPT] <<response>> [COMPLETION] "
    # and we want to train only on the completion.
    response_template_str = " <<" # This starts the prompt masking
    instruction_template_str = ">>" # This stops the instruction masking
    
    # Use tokenizer.encode to get IDs, excluding the first token if it's a space/BOS token
    l_id = tokenizer.encode(response_template_str, add_special_tokens=False) 
    r_id = tokenizer.encode(instruction_template_str, add_special_tokens=False) 


    collator = DataCollatorForCompletionOnlyLM(
        # response_template is the marker immediately preceding the text you want the model to learn.
        response_template=l_id, 
        # instruction_template is optional and specifies the start of the prompt, often used for instruction tuning.
        instruction_template=r_id, 
        tokenizer=tokenizer,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        args=training_args,
    )

    print(f"{trainer.model}")
    model.print_trainable_parameters() # Moved this here for proper display before training start

    print("\n🚀 Starting training...")
    trainer.train(resume_from_checkpoint=None)

    # Save the model
    print("\n💾 Saving model with the current parameters")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir) # Add tokenizer save
    print("✓ Training complete!")


# --- HYPERPARAMETER DEFINITION ---

model_args = ModelArguments(
    model_name_or_path="marcelbinz/Llama-3.1-Centaur-70B-adapter",
    lora_r=8,
    lora_alpha=8,
    lora_dropout=0,
)

data_args = DataTrainingArguments(
    dataset_text_field="text",
    max_seq_length=max_seq_length,
)

training_args = TrainingArguments(
    seed=100,
    num_train_epochs=num_train_epochs,
    log_level="info",
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_steps=100,
    eval_strategy ="steps",
    eval_steps= 100,
    learning_rate=learning_rate,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    weight_decay=weight_decay,
    warmup_steps=10,
    output_dir=f"{preffix_folder}centaur-ptsd-finetuned_{learning_rate}_{max_seq_length}_{train_batch_size}_{eval_batch_size}_{gradient_accumulation_steps}_{weight_decay}_{num_train_epochs}",
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    deepspeed="ds_config.json",
    fp16=False,
    bf16=True, # Recommended for Llama 3.1 on modern GPUs
    remove_unused_columns=False,
)

# START TRAINING!
print("\n" + "=" * 60)
print("🚀 STARTING TRAINING WITH ORIGINAL CENTAUR PARAMETERS")
print(f"Base model: {model_args.model_name_or_path}")
print(f"Effective batch size (per GPU): {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Learning rate: {training_args.learning_rate}")
print("=" * 60 + "\n")


# this is the main call
main(model_args, data_args, training_args)


print("Centaur full build run - Training completed!")

# --- PLOTTING SECTION ---

output_dir = f"{preffix_folder}centaur-ptsd-finetuned_{learning_rate}_{max_seq_length}_{train_batch_size}_{eval_batch_size}_{gradient_accumulation_steps}_{weight_decay}_{num_train_epochs}"
trainer_state_path = os.path.join(output_dir, "trainer_state.json")

if not os.path.exists(output_dir):
    print(f"Error: Output directory '{output_dir}' not found.")
else:
    print(f"Contents of '{output_dir}':")
    for item in os.listdir(output_dir):
        print(item)

    if not os.path.exists(trainer_state_path):
        print(f"Error: trainer_state.json not found at {trainer_state_path}")
        checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith('checkpoint-')]

        found_in_checkpoint = False
        for cp_dir in checkpoint_dirs:
            potential_path = os.path.join(output_dir, cp_dir, "trainer_state.json")
            if os.path.exists(potential_path):
                trainer_state_path = potential_path
                print(f"Found trainer_state.json in: {trainer_state_path}")
                found_in_checkpoint = True
                break

        if not found_in_checkpoint:
            print("Could not find trainer_state.json in any checkpoint subdirectory either.")

    if os.path.exists(trainer_state_path):
        try:
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)

            log_history = trainer_state.get('log_history', [])

            train_losses = []
            steps = []
            for log_entry in log_history:
                if 'loss' in log_entry and 'step' in log_entry:
                    train_losses.append(log_entry['loss'])
                    steps.append(log_entry['step'])

            if train_losses and steps:
                plt.figure(figsize=(10, 6))
                plt.plot(steps, train_losses, marker='o', linestyle='-', color='b')
                plt.title('Training Loss Over Steps')
                plt.xlabel('Training Steps')
                plt.ylabel('Loss')
                plt.grid(True)
                plot_filename = f"loss_plot_{learning_rate}_{max_seq_length}_{train_batch_size}_{eval_batch_size}_{gradient_accumulation_steps}_{weight_decay}_{num_train_epochs}.jpg"
                plt.savefig(plot_filename)
                print(f"Saved loss plot to {plot_filename}")
            else:
                print("No training loss data found in log_history.")
        except Exception as e:
            print(f"An error occurred while loading or plotting trainer_state.json: {e}")

