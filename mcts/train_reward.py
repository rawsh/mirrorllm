from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy

import random
from collections import Counter

# Define and parse arguments.
@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(
        default=-1, metadata={"help": "Used for multi-gpu"}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=32)
    # learning_rate: Optional[float] = field(default=2e-6)
    # embedding_learning_rate: Optional[float] = field(default=1e-6)
    learning_rate: Optional[float] = field(default=1e-5)
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="google/gemma-2-2b",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1,
        # default=3,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    train_set_path: Optional[str] = field(
        default="rawsh/magpie-ultra-v0.1-PRM-data-base",
        metadata={"help": "The dir of the subset of the training data to use"},
    )
    eval_set_path: Optional[str] = field(
        default="rawsh/magpie-ultra-v0.1-PRM-data-base",
        metadata={"help": "The dir of the subset of the eval data to use"},
    )
    output_path: Optional[str] = field(
        default="./mirrorgemma-2-2b-prm-base",
        metadata={"help": "The dir for output model"},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_torch_fused",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=8192)
    save_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Save the model every x steps"},
    )
    eval_every_steps: Optional[int] = field(
        default=999999,
        metadata={"help": "Eval the model every x steps"},
    )

def build_dataset(tokenizer, train_path, eval_path):
    def tokenize(sample):
        question = sample['question']
        steps = sample['steps']
        final_step_reward = sample['final_step_reward']

        formatted_steps = "\n\n".join(steps)
        full_text = f"{question}\n\n{formatted_steps}"

        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )

        sample["input_ids"] = tokenized["input_ids"]
        sample["attention_mask"] = tokenized["attention_mask"]
        sample["reward"] = final_step_reward
        return sample

    # Load and shuffle the training dataset
    ds_train = load_dataset(train_path, split="train").shuffle(seed=42)
    ds_train = ds_train.map(tokenize, num_proc=24)

    # Step 2: Assign bin number to each sample in training data
    def assign_bin(example):
        final_step_reward = example['final_step_reward']
        # Calculate bin number (bins: 0.0-0.1 => bin 0, ..., 0.9-1.0 => bin 9)
        bin_number = int(final_step_reward * 10)
        if bin_number == 10:
            bin_number = 9  # Handle the edge case where final_step_reward == 1.0
        example['bin'] = bin_number
        return example

    ds_train = ds_train.map(assign_bin, num_proc=24)

    # Step 3: Get counts of samples in each bin for training data
    bin_counts_train = Counter(ds_train['bin'])
    print("Training bin counts before undersampling:", bin_counts_train)

    # Determine the minimum count across all bins in training data
    min_count_train = min(bin_counts_train.values())
    print("Training minimum count per bin:", min_count_train)

    # Step 4: Create a mapping from bin to indices for training data
    bin_to_indices_train = {i: [] for i in range(10)}  # Bins 0 to 9
    for idx, bin_number in enumerate(ds_train['bin']):
        bin_to_indices_train[bin_number].append(idx)

    # Randomly sample min_count_train indices per bin for training data
    random.seed(42)
    selected_indices_train = []
    for bin_number, indices in bin_to_indices_train.items():
        if len(indices) >= min_count_train:
            sampled_indices = random.sample(indices, min_count_train)
        else:
            sampled_indices = indices  # Keep all samples if less than min_count_train
        selected_indices_train.extend(sampled_indices)

    # Shuffle the selected indices to mix the data
    random.shuffle(selected_indices_train)

    # Step 5: Create the balanced training dataset
    train_dataset = ds_train.select(selected_indices_train)
    print("Total training samples after undersampling:", len(train_dataset))

    # Now, build the evaluation dataset
    # Load and shuffle the evaluation dataset
    ds_eval = load_dataset(eval_path, split="train").shuffle(seed=42)
    ds_eval = ds_eval.map(tokenize, num_proc=24)

    # Assign bins to evaluation data
    ds_eval = ds_eval.map(assign_bin, num_proc=24)

    # Get counts of samples in each bin for evaluation data
    bin_counts_eval = Counter(ds_eval['bin'])
    print("Evaluation bin counts before undersampling:", bin_counts_eval)

    # Determine the minimum count per bin for evaluation data
    # Set it to be 10% of min_count_train, at least 1
    eval_min_count_per_bin = max(1, int(min_count_train * 0.1))
    print("Evaluation minimum count per bin:", eval_min_count_per_bin)

    # Create a mapping from bin to indices for evaluation data
    bin_to_indices_eval = {i: [] for i in range(10)}  # Bins 0 to 9
    for idx, bin_number in enumerate(ds_eval['bin']):
        bin_to_indices_eval[bin_number].append(idx)

    # Randomly sample eval_min_count_per_bin indices per bin for evaluation data
    selected_indices_eval = []
    for bin_number, indices in bin_to_indices_eval.items():
        if len(indices) >= eval_min_count_per_bin:
            sampled_indices = random.sample(indices, eval_min_count_per_bin)
        else:
            sampled_indices = indices  # Keep all samples if less than eval_min_count_per_bin
        selected_indices_eval.extend(sampled_indices)

    # Shuffle the selected indices to mix the data
    random.shuffle(selected_indices_eval)

    # Create the balanced evaluation dataset
    eval_dataset = ds_eval.select(selected_indices_eval)
    print("Total evaluation samples after undersampling:", len(eval_dataset))

    return train_dataset, eval_dataset

@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = [{
            "input_ids": feature["input_ids"],
            "attention_mask": feature["attention_mask"],
        } for feature in features]
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "rewards": torch.tensor([feature["reward"] for feature in features], dtype=torch.float),
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }
        return batch

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions.squeeze()
    labels = eval_pred.label_ids
    mse = np.mean((predictions - labels) ** 2)
    return {"mse": mse}

class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0].squeeze()
        loss = nn.functional.mse_loss(rewards, inputs["rewards"])
        
        if return_outputs:
            return loss, {"rewards": rewards}
        return loss

def train_reward_model():
    # Hardcode args (or you can parse arguments)
    script_args = ScriptArguments()

    # Load the model and tokenizer
    tokenizer_name = script_args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_auth_token=True)

    # Adjusted according to the base model
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = script_args.max_length

    # Get the datasets
    train_path = script_args.train_set_path
    eval_path = script_args.eval_set_path
    output_name = script_args.output_path

    train_dataset, eval_dataset = build_dataset(tokenizer, train_path, eval_path)
    print("Training set size:", len(train_dataset))
    print("Evaluation set size:", len(eval_dataset))

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=output_name,
        learning_rate=script_args.learning_rate,
        # embedding_learning_rate=script_args.embedding_learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_every_steps,
        save_strategy="steps",
        save_steps=script_args.save_every_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=script_args.bf16,
        logging_strategy="steps",
        logging_steps=10,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=0.03,
        report_to='wandb',
        torch_compile=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
    )

    model.config.use_cache = not script_args.gradient_checkpointing

    # Initialize the trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, max_length=script_args.max_length
        ),
    )

    # Start training
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.save_model(output_name + "/last_checkpoint")
    tokenizer.save_pretrained(output_name + "/last_checkpoint")

    # Push the model to Hugging Face Hub
    # Ensure you have the necessary permissions and authentication
    trainer.push_to_hub("rawsh/mirrorgemma-2-2b-PRM-base")

if __name__ == "__main__":
    train_reward_model()
