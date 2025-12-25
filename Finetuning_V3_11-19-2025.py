"""
Centaur-PTSD LoRA finetuning + logging script (optional). Using the combined files
from information from the Stroop task and Fear Conditional task - only few subjects added until now..

This script:
- Parses hyperparameters from the command line
  (learning rate, max_seq_length, batch sizes, grad accumulation, weight decay, epochs).
- Downloads private JSONL train/test splits from Hugging Face
  (cognition PTSD task datasets) and enforces a fixed schema.
- Loads the `marcelbinz/Llama-3.1-Centaur-70B-adapter` base model with Unsloth,
  enabling 4-bit loading and LoRA adapters on attention/MLP blocks.
- Trains the model with `UnslothTrainer` using a completion-only LM collator
  (with <<response>> style templates) and saves checkpoints to a scratch directory
  named by the current hyperparameters.
- Optionally reloads the finetuned model into a `transformers.pipeline`
  for quick text-generation tests.
- Searches for `trainer_state.json` in the output/checkpoint folders and,
  if found, parses `log_history` to plot and save training loss vs. steps.we
"""


from transformers import TrainingArguments
import unsloth
import transformers
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.data import data_collator
from unsloth import is_bfloat16_supported, UnslothTrainer, UnslothTrainingArguments, FastLanguageModel
from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Features, Value
import json
import matplotlib.pyplot as plt

from huggingface_hub import hf_hub_download, notebook_login

"""
 define first the inputs as hyperparameter tunings inputs use them as
 python Finetuning_V3.py <learning_rate> <max_seq_length> <train_batch_size> <eval_batch_size> <gradient_accumulator> <weight_decay> <num_train_epochs>
 python Finetuning_V3.py 0.0001 32768 1 1 8 0.01 5 - as an example
"""

# --- REPOSITORY AND FILE CONSTANTS ---
TRAIN_REPO_ID = "arshad101/cognition_task_psych"
TRAIN_FILE_NAME = "combined_train_fixed.jsonl"

# -- set here the preffixed path for saving the model checkpoints
preffix_folder = "/scratch/axs7716Arsh"

# -- get the parameter here
learning_rate = float(sys.argv[1])
max_seq_length = int(sys.argv[2])
train_batch_size = int(sys.argv[3])
eval_batch_size = int(sys.argv[4])
gradient_accumulation_steps = int(sys.argv[5])
weight_decay = float(sys.argv[6])
num_train_epochs = int(sys.argv[7])

# Define the mandatory schema to force consistent data types
# This fixes the "Column(/participant) changed from string to number" error.
custom_features = Features({
    'text': Value('string'),
    'experiment': Value('string'),
    # <-- FIX for ArrowInvalid (forces string/text type)
    'participant': Value('string'),
    # IMPORTANT: Add ALL other column names from your JSONL file here!
    # e.g., 'response': Value('string'), 'score': Value('float')
})

# --- DATASET LOADING: Robust Private File Access ---

# 1. Download the private file to the local cache using authentication
train_file_path = hf_hub_download(
    repo_id=TRAIN_REPO_ID,
    filename=TRAIN_FILE_NAME,
    repo_type="dataset",
    token=True  # Uses the token from your notebook_login() session
)


# 2. Load dataset from the local path, using the JSON builder and the
# fixed schema.

# Training Data Load
raw_train_dataset = load_dataset(
    'json',
    # Map the local file path to the 'train' split
    data_files={'train': train_file_path},
    features=custom_features,  # Apply the fixed schema
)

train_dataset = raw_train_dataset['train'].shuffle()


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"})
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0)


# define the class and max_seq_length paramater from here
@dataclass
class DataTrainingArguments:
    dataset_text_field: str = field(default="text")
    max_seq_length: Optional[int] = field(default=max_seq_length)


def main(model_args, data_args, training_args):
    set_seed(training_args.seed)

    # Define the mandatory schema to force consistent data types
    custom_features = Features({
        'text': Value('string'),
        'experiment': Value('string'),
        'participant': Value('string'),
    })

    # --- REPOSITORY AND FILE CONSTANTS ---
    # this will be the configuration for Arshad account
    TRAIN_REPO_ID = "arshad101/cognition_task_psych"
    TRAIN_FILE_NAME = "combined_train_fixed.jsonl"
    EVAL_REPO_ID = "arshad101/cognition_task_psych_test"
    EVAL_FILE_NAME = "combined_test_fixed.jsonl"

    # 1. Download the private files to the local cache using authentication
    train_file_path = hf_hub_download(
        repo_id=TRAIN_REPO_ID,
        filename=TRAIN_FILE_NAME,
        repo_type="dataset",
        token=True  # Uses the token from your notebook_login() session
    )
    eval_file_path = hf_hub_download(
        repo_id=EVAL_REPO_ID,
        filename=EVAL_FILE_NAME,
        repo_type="dataset",
        token=True
    )

    # 2. Load datasets from local paths, using the JSON builder and the fixed
    # schema.
    raw_train_dataset = load_dataset(
        'json',
        data_files={'train': train_file_path},
        features=custom_features,
    )
    train_dataset = raw_train_dataset['train'].shuffle()

    raw_eval_dataset = load_dataset(
        'json',
        data_files={'test': eval_file_path},
        features=custom_features,
    )
    eval_dataset = raw_eval_dataset['test'].shuffle()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=data_args.max_seq_length,
        # dtype= "float16",
        load_in_4bit=True,
        # device_map="cuda"
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj",
            "up_proj", "down_proj",
        ],
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias='none',
        use_gradient_checkpointing="unsloth",
        random_state=training_args.seed,
        use_rslora=True,
        loftq_config=None

    )

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    l_id = tokenizer(" <<").input_ids[1:]
    r_id = tokenizer(">>").input_ids[1:]

    collator = DataCollatorForCompletionOnlyLM(
        response_template=l_id,
        instruction_template=r_id,
        tokenizer=tokenizer,

    )

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=data_args.max_seq_length,
        dataset_num_proc=8,
        data_collator=collator,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=training_args.per_device_train_batch_size,
            per_device_eval_batch_size=training_args.per_device_eval_batch_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            warmup_steps=training_args.warmup_steps,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            embedding_learning_rate=training_args.learning_rate / 10,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            log_level=training_args.log_level,
            logging_strategy=training_args.logging_strategy,
            logging_steps=training_args.logging_steps,
            # evaluation_strategy = training_args.evaluation_strategy,
            # eval_steps = training_args.eval_steps,
            save_strategy=training_args.save_strategy,
            save_steps=training_args.save_steps,
            optim=training_args.optim,
            weight_decay=training_args.weight_decay,
            lr_scheduler_type=training_args.lr_scheduler_type,
            seed=training_args.seed,
            output_dir=training_args.output_dir,
        ),

    )

    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()

    # Start training!
    print("\n🚀 Starting training...")
    trainer.train(resume_from_checkpoint=None)

    # Save the model
    print("\n💾 Saving model..with the current parameters")
    trainer.save_model()
    print("✓ Training complete!")


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
    # evaluation_strategy="steps",
    # eval_steps=999999,
    save_strategy="steps",
    save_steps=100,
    learning_rate=learning_rate,
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    weight_decay=weight_decay,
    warmup_steps=10,  # 100
    output_dir=f"{preffix_folder}centaur-ptsd-finetuned_{learning_rate}_{max_seq_length}_{train_batch_size}_{eval_batch_size}_{gradient_accumulation_steps}_{weight_decay}_{num_train_epochs}",
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
)

# START TRAINING!
print("\n" + "=" * 60)
print("🚀 STARTING TRAINING WITH ORIGINAL CENTAUR PARAMETERS")
# print("="*60)
print(f"Base model: {model_args.model_name_or_path}")
print(f"Effective batch size: {training_args.gradient_accumulation_steps}")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Learning rate: {training_args.learning_rate}")
print("=" * 60 + "\n")


# this is the main call
main(model_args, data_args, training_args)


print("Centaur full build run")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='marcelbinz/LLama-3.1-Centaur-70B-adapter',
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

print("Model Loaded .. Creating Pipeline")

pipe = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    pad_token_id=0,
    do_sample=True,
    temperature=1.0,
    max_new_tokens=1,
)


# uncomment this only in case you need to upload the model in huggingface
# we can save the model in the scratch directory - instead to upload so
# quick to huggingface

"""
from unsloth import FastLanguageModel
from huggingface_hub import login

  # Step 1: Load trained model
print("Loading trained model from local directory...")
model, tokenizer = FastLanguageModel.from_pretrained(
      model_name=f"centaur-ptsd-finetuned",  # Local output directory
      max_seq_length=32768,
      dtype=None,
      load_in_4bit=True,
  )
print("✓ Model loaded\n")

  # Step 2: Login to HuggingFace
print("Logging into HuggingFace...")
login()  # You'll be prompted for your token
print("✓ Logged in\n")

## ** use this only if you need to save the model we can leave it in scratch and save it
## to a main hardrive in case it is necessary
# Step 3: Upload model
print("Uploading model to HuggingFace Hub...")
model.push_to_hub(
      "arshad101/Centaur-PTSD-70B-LoRA",
      token=True,
      commit_message="Fine-tuned Centaur on PTSD cognitive tasks"
  )
print("✓ Model uploaded\n")

  # Step 4: Upload tokenizer
print("Uploading tokenizer...")
tokenizer.push_to_hub(
      "arshad101/Centaur-PTSD-70B-LoRA",
      token=True,
  )
print("✓ Tokenizer uploaded\n")

  # Success message
print("="*60)
print("SUCCESS! Model is now available at:")
print("https://huggingface.co/arshad101/Centaur-PTSD-70B-LoRA")
print("\nLoad it anytime with:")
print('model_name="arshad101/Centaur-PTSD-70B-LoRA"')
print("="*60)
"""

# DEFINE THIS AS THE PLOTTING SECTION..YOU CAN ALSO ADD IT IN A SEPARATE FILE**

# Define the path to the trainer_state.json file
output_dir = f"{preffix_folder}centaur-ptsd-finetuned_{learning_rate}_{max_seq_length}_{train_batch_size}_{eval_batch_size}_{gradient_accumulation_steps}_{weight_decay}_{num_train_epochs}"
trainer_state_path = os.path.join(output_dir, "trainer_state.json")

# Check if the output directory exists
if not os.path.exists(output_dir):
    print(
        f"Error: Output directory '{output_dir}' not found. The trainer_state.json file is inside ")
else:
    print(f"Contents of '{output_dir}':")
    for item in os.listdir(output_dir):
        print(item)

    # Now, try to load the trainer state if the file exists
    if not os.path.exists(trainer_state_path):
        print(f"Error: trainer_state.json not found at {trainer_state_path}")
        # Check common subdirectories for trainer_state.json, e.g., checkpoint
        checkpoint_dirs = [d for d in os.listdir(
            output_dir) if d.startswith('checkpoint-')]
        for cp_dir in checkpoint_dirs:
            potential_path = os.path.join(
                output_dir, cp_dir, "trainer_state.json")
            if os.path.exists(potential_path):
                trainer_state_path = potential_path
                print(f"Found trainer_state.json in: {trainer_state_path}")
                break

        if not os.path.exists(trainer_state_path):
            print(
                "Could not find trainer_state.json in any checkpoint subdirectory either.")

    if os.path.exists(trainer_state_path):
        # Load the trainer state
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)

        # Extract log history
        log_history = trainer_state.get('log_history', [])

        # Filter for training loss entries
        train_losses = []
        steps = []
        for log_entry in log_history:
            if 'loss' in log_entry and 'step' in log_entry:
                train_losses.append(log_entry['loss'])
                steps.append(log_entry['step'])

        # Plot the loss function
        if train_losses and steps:
            plt.figure(figsize=(10, 6))
            plt.plot(steps, train_losses, marker='o', linestyle='-', color='b')
            plt.title('Training Loss Over Steps')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(f"loss_plot_{learning_rate}_{max_seq_length}_{train_batch_size}_{eval_batch_size}_{gradient_accumulation_steps}_{weight_decay}_{num_train_epochs}.jpg")
        else:
            print("No training loss data found in log_history.")
