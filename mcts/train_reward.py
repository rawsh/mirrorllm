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
    EarlyStoppingCallback,
)
from transformers.utils import PaddingStrategy
from huggingface_hub import HfFolder
import os
import random
from collections import Counter

@dataclass
class ScriptArguments:
    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    deepspeed: Optional[str] = field(default=None)
    per_device_train_batch_size: Optional[int] = field(default=8)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=32)
    learning_rate: Optional[float] = field(default=1e-5)
    weight_decay: Optional[float] = field(default=0.0001)
    model_name: Optional[str] = field(default="Qwen/Qwen2.5-0.5B")
    model_revision: Optional[str] = field(default=None)
    bf16: Optional[bool] = field(default=True)
    num_train_epochs: Optional[int] = field(default=2)
    train_set_path: Optional[str] = field(default="rawsh/magpie-ultra-v0.1-PRM-data-base")
    eval_set_path: Optional[str] = field(default="rawsh/magpie-ultra-v0.1-PRM-data-base")
    output_path: Optional[str] = field(default="./mirrorqwen2.5-0.5b-prm-base")
    output_model_name: Optional[str] = field(default="rawsh/mirrorqwen2.5-0.5b-prm")
    gradient_checkpointing: Optional[bool] = field(default=True)
    optim: Optional[str] = field(default="adamw_torch_fused")
    lr_scheduler_type: Optional[str] = field(default="cosine")
    max_length: Optional[int] = field(default=8192)
    save_every_steps: Optional[int] = field(default=999999)
    eval_every_steps: Optional[int] = field(default=999999)
    early_stopping_patience: Optional[int] = field(default=3)
    early_stopping_threshold: Optional[float] = field(default=0.001)
    disable_binning: Optional[bool] = field(default=False)
    warmup_steps: Optional[int] = field(default=100)
    save_total_limit: Optional[int] = field(default=3)
    min_loss_threshold: Optional[float] = field(default=0.1)
    hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace Hub token. If not provided, will try to use the cached token."}
    )
    push_to_hub: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to push the model to the HuggingFace Hub"}
    )

def build_dataset(tokenizer, train_path, eval_path, disable_binning: bool):
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

    if not disable_binning:
        # Step 2: Assign bin number to each sample in training data
        def assign_bin(example):
            final_step_reward = example['final_step_reward']
            if final_step_reward <= 0.1:
                # Samples with rewards <= 0.1 get assigned to bin -1 (won't be balanced)
                example['bin'] = -1
            else:
                # Calculate bin number for rewards > 0.1 (bins: 0.1-0.2 => bin 0, ..., 0.9-1.0 => bin 8)
                bin_number = int((final_step_reward - 0.1) * 10)
                if bin_number == 9:
                    bin_number = 8  # Handle the edge case where final_step_reward == 1.0
                example['bin'] = bin_number
            return example

        ds_train = ds_train.map(assign_bin, num_proc=24)

        # Step 3: Separate low reward samples and get counts for other bins
        low_reward_indices = [idx for idx, bin_num in enumerate(ds_train['bin']) if bin_num == -1]
        bin_counts_train = Counter([b for b in ds_train['bin'] if b >= 0])
        print("Training bin counts before undersampling (excluding ≤0.1):", bin_counts_train)

        # Determine the minimum count across all bins in training data (excluding bin -1)
        min_count_train = min(bin_counts_train.values()) if bin_counts_train else 0
        print("Training minimum count per bin (excluding ≤0.1):", min_count_train)

        # Step 4: Create a mapping from bin to indices for training data
        bin_to_indices_train = {i: [] for i in range(9)}  # Bins 0 to 8 (for rewards 0.1-1.0)
        for idx, bin_number in enumerate(ds_train['bin']):
            if bin_number >= 0:  # Only process samples with rewards > 0.1
                bin_to_indices_train[bin_number].append(idx)

        # Randomly sample min_count_train indices per bin for training data
        random.seed(42)
        selected_indices_train = []
        # First add all low reward samples (≤0.1)
        selected_indices_train.extend(low_reward_indices)
        # Then sample from other bins
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
        print("Total training samples after processing:", len(train_dataset))
        print("- Samples with reward ≤0.1:", len(low_reward_indices))
        print("- Samples per bin >0.1:", min_count_train)
    else:
        train_dataset = ds_train

    # Now, build the evaluation dataset similarly
    ds_eval = load_dataset(eval_path, split="train").shuffle(seed=42)
    ds_eval = ds_eval.map(tokenize, num_proc=24)

    if not disable_binning:
        # Assign bins to evaluation data
        ds_eval = ds_eval.map(assign_bin, num_proc=24)

        # Separate low reward samples and get counts for other bins
        eval_low_reward_indices = [idx for idx, bin_num in enumerate(ds_eval['bin']) if bin_num == -1]
        bin_counts_eval = Counter([b for b in ds_eval['bin'] if b >= 0])
        print("Evaluation bin counts before undersampling (excluding ≤0.1):", bin_counts_eval)

        # Determine the minimum count per bin for evaluation data
        # Set it to be 10% of min_count_train, at least 1
        eval_min_count_per_bin = max(1, int(min_count_train * 0.1))
        print("Evaluation minimum count per bin (excluding ≤0.1):", eval_min_count_per_bin)

        # Create a mapping from bin to indices for evaluation data
        bin_to_indices_eval = {i: [] for i in range(9)}  # Bins 0 to 8
        for idx, bin_number in enumerate(ds_eval['bin']):
            if bin_number >= 0:  # Only process samples with rewards > 0.1
                bin_to_indices_eval[bin_number].append(idx)

        # Randomly sample eval_min_count_per_bin indices per bin for evaluation data
        selected_indices_eval = []
        # First add all low reward samples (≤0.1)
        selected_indices_eval.extend(eval_low_reward_indices)
        # Then sample from other bins
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
        print("Total evaluation samples after processing:", len(eval_dataset))
        print("- Evaluation samples with reward ≤0.1:", len(eval_low_reward_indices))
        print("- Evaluation samples per bin >0.1:", eval_min_count_per_bin)
    else:
        eval_dataset = ds_eval

    return train_dataset, eval_dataset

@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
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
    """
    Compute metrics for the model evaluation.
    Args:
        eval_pred: tuple of predictions and labels
    Returns:
        dict: Dictionary containing metrics
    """
    predictions = eval_pred.predictions.squeeze()
    labels = eval_pred.label_ids
    mse = np.mean((predictions - labels) ** 2)
    return {
        "mse": mse,
        "mse_moving_avg": mse  # Just use MSE directly since we're serverless
    }

class RewardTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_steps = self.args.warmup_steps
        self.current_step = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = model(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"]
        )[0].squeeze()
        loss = nn.functional.mse_loss(rewards, inputs["rewards"])
        
        if return_outputs:
            return loss, {"rewards": rewards}
        return loss

    def _push_to_hub(self, commit_message: Optional[str] = "End of training", **kwargs) -> str:
        """Override to force push to hub by using the low-level API"""
        if not self.args.push_to_hub:
            return

        if not self.args.hub_token and not HfFolder.get_token():
            raise ValueError(
                "No token provided and no token found in cache. "
                "Please provide a token via hub_token parameter or log in using `huggingface-cli login`"
            )

        # Set token if provided
        token = self.args.hub_token or HfFolder.get_token()
        if token:
            os.environ["HF_TOKEN"] = token
            HfFolder.save_token(token)

        try:
            from huggingface_hub import HfApi, create_repo
            api = HfApi()
            
            # Create or ensure repo exists
            repo_id = self.args.hub_model_id
            try:
                create_repo(repo_id, token=token, exist_ok=True, private=True)
                print(f"Repository {repo_id} is ready")
            except Exception as e:
                print(f"Note: {str(e)}")

            # Save model and tokenizer locally first
            local_path = os.path.join(self.args.output_dir, "for_hub_push")
            os.makedirs(local_path, exist_ok=True)
            self.save_model(local_path)
            
            # Use upload_folder with low-level API
            api.upload_folder(
                folder_path=local_path,
                repo_id=repo_id,
                token=token,
                repo_type="model",
                commit_message=commit_message,
                revision="main",
                create_pr=False,
            )
            print(f"Successfully uploaded folder to {repo_id}")
            
            return repo_id
            
        except Exception as e:
            print(f"Error in push_to_hub: {str(e)}")
            raise e

def train_reward_model(
    model_name=None,
    model_revision=None,
    dataset_path=None,
    output_model_name=None,
    disable_binning=False,
    hub_token=None,
    push_to_hub=True
):
    script_args = ScriptArguments(
        disable_binning=disable_binning,
        warmup_steps=10,
        save_total_limit=3,
        min_loss_threshold=0.1,
        hub_token=hub_token,
        push_to_hub=push_to_hub
    )

    if model_name:
        script_args.model_name = model_name
    if model_revision:
        script_args.model_revision = model_revision
    if output_model_name:
        script_args.output_model_name = output_model_name
    if dataset_path:
        script_args.train_set_path = dataset_path
        script_args.eval_set_path = dataset_path

    # Initialize tokenizer with better error handling
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_name,
            use_auth_token=True if script_args.hub_token else None,
            revision=script_args.model_revision
        )
        tokenizer.truncation_side = "left"
        tokenizer.model_max_length = script_args.max_length
        tokenizer.pad_token = tokenizer.eos_token

    except Exception as e:
        raise RuntimeError(f"Failed to initialize tokenizer: {str(e)}")

    # Enhanced training arguments with Hub settings
    training_args = TrainingArguments(
        output_dir=script_args.output_path,
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        weight_decay=script_args.weight_decay,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        deepspeed=script_args.deepspeed,
        local_rank=script_args.local_rank,
        remove_unused_columns=False,
        label_names=[],
        bf16=script_args.bf16,
        logging_strategy="steps",
        logging_steps=1,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=0.05,
        report_to='wandb',
        torch_compile=True,
        load_best_model_at_end=True,
        metric_for_best_model="mse_moving_avg",
        greater_is_better=False,
        save_strategy="steps",
        save_steps=max(5, script_args.eval_every_steps),
        evaluation_strategy="steps",
        eval_steps=max(5, script_args.eval_every_steps),
        save_total_limit=script_args.save_total_limit,
        max_grad_norm=1.0,
        # Hub-specific settings
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.output_model_name if script_args.push_to_hub else None,
        hub_token=script_args.hub_token,
    )

    # Initialize model with better error handling
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            script_args.model_name,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            use_flash_attention_2=True,
            use_auth_token=True if script_args.hub_token else None,
            revision=script_args.model_revision
        )
        model.config.pad_token_id = model.config.eos_token_id
        model.config.use_cache = not script_args.gradient_checkpointing
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {str(e)}")

    # Build datasets
    train_dataset, eval_dataset = build_dataset(
        tokenizer, 
        script_args.train_set_path, 
        script_args.eval_set_path, 
        script_args.disable_binning
    )

    # Initialize trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, 
            max_length=script_args.max_length
        ),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=script_args.early_stopping_patience,
                early_stopping_threshold=script_args.early_stopping_threshold
            )
        ],
    )

    # Train and save
    trainer.train()
    
    print("Saving final checkpoint")
    trainer.save_model(script_args.output_path)
    tokenizer.save_pretrained(script_args.output_path)
    
    # Push to Hub with force push handling
    if script_args.push_to_hub and script_args.output_model_name:
        try:
            print(f"Pushing model to hub: {script_args.output_model_name}")
            # First try to push the tokenizer separately with force
            try:
                tokenizer.push_to_hub(
                    script_args.output_model_name,
                    use_auth_token=script_args.hub_token,
                    force=True
                )
                print("Successfully pushed tokenizer to hub!")
            except Exception as e:
                print(f"Warning: Failed to push tokenizer: {str(e)}")

            # Then push the model with force
            trainer.push_to_hub(
                commit_message="Final trained model",
                blocking=True,  # Wait for push to complete
                force=True  # Force the push
            )
            print("Successfully pushed model to hub!")
        except Exception as e:
            print(f"Error pushing to hub: {str(e)}")
            print("Saving model locally anyway...")

if __name__ == "__main__":
    train_reward_model()