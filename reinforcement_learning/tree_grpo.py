"""
Implementation of Group Relative Policy Optimization (GRPO).
Extends TRL's functionality without modifying source code.
"""

import gc
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedTokenizer,
    TrainerCallback,
    Trainer
)

from trl.trainer.utils import (
    # OnPolicyTrainer,
    disable_dropout_in_model,
    exact_div,
    get_reward,
    # masked_mean,
    # masked_whiten,
    prepare_deepspeed,
)

@dataclass 
class GRPOConfig:
    """Configuration for GRPO training."""
    exp_name: str = "grpo_training"
    """The name of this experiment"""
    
    reward_model_path: str = None 
    """Path to the reward model"""
    
    num_grpo_epochs: int = 4
    """Number of GRPO training epochs"""
    
    whiten_rewards: bool = False
    """Whether to whiten the rewards"""
    
    kl_coef: float = 0.05
    """KL divergence coefficient"""
    
    cliprange: float = 0.2
    """PPO-style clipping range"""
    
    use_process_supervision: bool = False
    """Enable process supervision"""
    
    use_iterative_reward_model_training: bool = True
    """Enable iterative reward model training"""
    
    sampling_group_size: int = 4
    """Number of samples to generate per input"""
    
    sampling_strategy: str = "top_p"
    """Sampling strategy: 'top_p' or 'top_k'"""
    
    sampling_strategy_top_p: float = 0.95
    """Top-p sampling parameter"""
    
    sampling_strategy_top_k: int = 0
    """Top-k sampling parameter"""
    
    sampling_temperature: float = 0.1
    """Temperature for sampling"""

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.sampling_strategy not in ["top_p", "top_k"]:
            raise ValueError("sampling_strategy must be 'top_p' or 'top_k'")
        
        if self.sampling_group_size < 1:
            raise ValueError("sampling_group_size must be positive")
        
        if self.sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be positive")
            
        if self.sampling_strategy_top_p <= 0 or self.sampling_strategy_top_p > 1:
            raise ValueError("sampling_strategy_top_p must be in (0, 1]")
            
        if self.sampling_strategy_top_k < 0:
            raise ValueError("sampling_strategy_top_k must be non-negative")

class GRPOTrainer(Trainer):
    """
    Implementation of Group Relative Policy Optimization (GRPO) trainer.
    Extends TRL's OnPolicyTrainer with GRPO-specific functionality.
    """

    def __init__(
        self,
        config: GRPOConfig,
        tokenizer: PreTrainedTokenizer,
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        """Initialize the GRPO trainer.

        Args:
            config: GRPO configuration
            tokenizer: Tokenizer for text processing
            policy: Policy model to be trained
            ref_policy: Reference policy model
            reward_model: Model for computing rewards
            train_dataset: Training dataset
            data_collator: Data collator
            eval_dataset: Evaluation dataset
            callbacks: Training callbacks
        """
        super().__init__(config, tokenizer)
        
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        
        # Set generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=config.response_length,
            min_new_tokens=config.response_length,
            temperature=config.sampling_temperature,
            top_k=config.sampling_strategy_top_k,
            top_p=config.sampling_strategy_top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Initialize models
        for model in [policy, ref_policy, reward_model]:
            disable_dropout_in_model(model)
            
        # Setup accelerator and devices
        self.accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
        
        # Configure batch sizes
        self.configure_batch_sizes()
        
        # Initialize dataloaders
        self.setup_dataloaders()
        
        # Prepare models
        self.prepare_models()

    def configure_batch_sizes(self):
        """Configure various batch sizes needed for training."""
        args = self.config
        
        # Calculate total batch size
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, 
            args.num_mini_batches,
            "`batch_size` must be a multiple of `num_mini_batches`"
        )
        
        # Local batch sizes
        args.local_mini_batch_size = exact_div(
            args.local_batch_size,
            args.num_mini_batches,
            "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        
        # Calculate number of total batches
        args.num_total_batches = math.ceil(
            args.total_episodes * args.sampling_group_size / args.batch_size
        )

    def setup_dataloaders(self):
        """Setup training and evaluation dataloaders."""
        # Training dataloader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.local_batch_size // self.config.sampling_group_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True
        )
        
        # Evaluation dataloader if dataset provided
        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=self.config.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                drop_last=True
            )
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)

    def prepare_models(self):
        """Prepare models for distributed training."""
        # Prepare main models
        self.policy, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.policy, self.optimizer, self.train_dataloader
        )
        
        # Handle deepspeed case
        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(self.reward_model, self.config)
            self.ref_policy = prepare_deepspeed(self.ref_policy, self.config)
            self.deepspeed = self.policy
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)

    def compute_advantages(self, rewards: torch.Tensor, sequence_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages for GRPO using either process or outcome supervision.
        
        Args:
            rewards: Reward tensor of shape (batch_size, sequence_length)
            sequence_lengths: Sequence length tensor of shape (batch_size,)
            
        Returns:
            Tensor of advantages with shape (batch_size, sequence_length)
        """
        if self.config.use_process_supervision:
            # Process supervision: compute advantages per step
            # Normalize rewards within group
            mean = rewards.mean(dim=0, keepdim=True)
            std = rewards.std(dim=0, keepdim=True) + 1e-8
            normalized_rewards = (rewards - mean) / std
            
            # Compute advantages as sum of future rewards
            advantages = torch.zeros_like(rewards)
            for t in range(sequence_lengths.max()):
                mask = sequence_lengths > t
                # Sum up future rewards for each position
                future_rewards = normalized_rewards[:, t:]  
                advantages[mask, t] = future_rewards[mask].sum(dim=1)
                
        else:
            # Outcome supervision: use final reward for all positions
            # Normalize rewards within group
            group_size = self.config.sampling_group_size
            rewards_reshaped = rewards.view(-1, group_size)
            mean = rewards_reshaped.mean(dim=1, keepdim=True)
            std = rewards_reshaped.std(dim=1, keepdim=True) + 1e-8
            normalized_rewards = (rewards_reshaped - mean) / std
            normalized_rewards = normalized_rewards.view(rewards.shape)
            
            # Broadcast final reward to all positions
            advantages = normalized_rewards.unsqueeze(1).expand(-1, sequence_lengths.max())
            
            # Mask padding positions
            padding_mask = torch.arange(advantages.size(1), device=advantages.device) >= sequence_lengths.unsqueeze(1)
            advantages = advantages.masked_fill(padding_mask, 0.0)
            
        return advantages
        
    def compute_kl_penalty(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        sequence_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence penalty between policy and reference model.
        
        Args:
            logprobs: Log probabilities from policy model
            ref_logprobs: Log probabilities from reference model
            sequence_lengths: Sequence length tensor
            
        Returns:
            KL divergence penalty tensor
        """
        # Compute unbiased KL estimator
        kl_div = (ref_logprobs.exp() * (ref_logprobs - logprobs)).mean(dim=-1)
        
        # Mask padding
        padding_mask = torch.arange(kl_div.size(1), device=kl_div.device) >= sequence_lengths.unsqueeze(1)
        kl_div = kl_div.masked_fill(padding_mask, 0.0)
        
        return self.config.kl_coef * kl_div
        
    def sample_sequences(self, query_tensors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample sequence groups from the policy model.
        
        Args:
            query_tensors: Input query tensors
            
        Returns:
            Tuple of (sampled sequences, logprobs, sequence lengths)
        """
        # Repeat queries for group sampling
        queries = query_tensors.repeat(self.config.sampling_group_size, 1) 
        
        sampling_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": True,
            "temperature": self.config.sampling_temperature
        }
        
        if self.config.sampling_strategy == "top_p":
            sampling_kwargs["top_p"] = self.config.sampling_strategy_top_p
        else:
            sampling_kwargs["top_k"] = self.config.sampling_strategy_top_k
            
        # Generate sequences
        outputs = self.policy.generate(
            queries,
            **sampling_kwargs,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # Extract logprobs and sequence lengths
        logprobs = torch.stack(outputs.scores, dim=1)
        logprobs = F.log_softmax(logprobs, dim=-1)
        
        sequences = outputs.sequences
        sequence_lengths = (sequences != self.tokenizer.pad_token_id).sum(dim=1)
        
        return sequences, logprobs, sequence_lengths

    def train(self):
        """Main training loop implementing GRPO algorithm."""
        args = self.config
        model = self.policy
        optimizer = self.optimizer
        
        # Training state initialization
        self.model_wrapped = model
        stats_shape = (args.num_grpo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        stats = {
            'approxkl': torch.zeros(stats_shape, device=self.accelerator.device),
            'pg_clipfrac': torch.zeros(stats_shape, device=self.accelerator.device),
            'pg_loss': torch.zeros(stats_shape, device=self.accelerator.device),
            'entropy': torch.zeros(stats_shape, device=self.accelerator.device),
            'ratio': torch.zeros(stats_shape, device=self.accelerator.device)
        }
        
        def repeat_dataloader():
            while True:
                yield from self.train_dataloader
                
        iter_dataloader = iter(repeat_dataloader())
        
        accelerator = self.accelerator
        device = accelerator.device
        
        accelerator.print("=== Starting GRPO Training ===")
        start_time = time.time()
        
        model.train()
        
        # Iterate over training batches
        for update in range(1, args.num_total_batches + 1):
            self.state.episode += args.batch_size
            
            # Get batch of queries
            data = next(iter_dataloader)
            queries = data["input_ids"].to(device)
            
            with torch.no_grad():
                # Sample sequences and compute rewards
                sequences, logprobs, sequence_lengths = self.sample_sequences(queries)
                rewards = self.compute_rewards(sequences)
                
                # Get reference model logprobs
                ref_outputs = self.ref_policy(sequences)
                ref_logits = ref_outputs.logits[:, :-1]
                ref_logprobs = F.log_softmax(ref_logits, dim=-1)
                
                # Compute advantages
                advantages = self.compute_advantages(rewards, sequence_lengths)
                
                # Store original sampling distribution for importance sampling
                old_logprobs = logprobs.detach()
            
            # GRPO training epochs
            for grpo_epoch in range(args.num_grpo_epochs):
                # Randomly permute batch for minibatches
                batch_indices = np.random.permutation(args.local_batch_size)
                
                minibatch_idx = 0
                for mb_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mb_end = mb_start + args.local_mini_batch_size
                    mb_indices = batch_indices[mb_start:mb_end]
                    
                    grad_accum_idx = 0
                    for ga_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        # Process microbatch with gradient accumulation
                        with accelerator.accumulate(model):
                            ga_end = ga_start + args.per_device_train_batch_size
                            microbatch_indices = mb_indices[ga_start:ga_end]
                            
                            # Get microbatch tensors
                            mb_seqs = sequences[microbatch_indices]
                            mb_old_logprobs = old_logprobs[microbatch_indices]
                            mb_advantages = advantages[microbatch_indices]
                            mb_ref_logprobs = ref_logprobs[microbatch_indices]
                            
                            # Forward pass
                            outputs = model(mb_seqs)
                            logits = outputs.logits[:, :-1]
                            new_logprobs = F.log_softmax(logits, dim=-1)
                            
                            # Policy loss with clipping
                            ratio = torch.exp(new_logprobs - mb_old_logprobs)
                            pg_losses = -mb_advantages * ratio
                            pg_losses2 = -mb_advantages * torch.clamp(
                                ratio,
                                1.0 - args.cliprange,
                                1.0 + args.cliprange
                            )
                            
                            # Add KL penalty
                            kl_penalty = self.compute_kl_penalty(
                                new_logprobs,
                                mb_ref_logprobs,
                                sequence_lengths[microbatch_indices]
                            )
                            
                            # Final loss
                            pg_loss = torch.max(pg_losses, pg_losses2).mean()
                            loss = pg_loss + kl_penalty.mean()
                            
                            # Backward pass
                            accelerator.backward(loss)
                            if args.max_grad_norm is not None:
                                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()
                            
                            # Update stats
                            with torch.no_grad():
                                stats['approxkl'][grpo_epoch, minibatch_idx, grad_accum_idx] = 0.5 * ((new_logprobs - mb_old_logprobs) ** 2).mean()
                                stats['pg_clipfrac'][grpo_epoch, minibatch_idx, grad_accum_idx] = (pg_losses2 > pg_losses).float().mean()
                                stats['pg_loss'][grpo_epoch, minibatch_idx, grad_accum_idx] = pg_loss
                                stats['entropy'][grpo_epoch, minibatch_idx, grad_accum_idx] = -new_logprobs.mean()
                                stats['ratio'][grpo_epoch, minibatch_idx, grad_accum_idx] = ratio.mean()
                                
                        grad_accum_idx += 1
                    minibatch_idx += 1
                    
                    # Clean up 
                    del outputs, logits, new_logprobs, ratio, pg_losses, pg_losses2, loss
                    torch.cuda.empty_cache()

            # Log metrics
            self.log_metrics(stats, update, start_time)
            
            # Update learning rate
            self.lr_scheduler.step()
            
            # Save checkpoint if needed
            if self.should_save(update):
                self.save_model()
                
            # Run evaluation if needed
            if self.should_evaluate(update):
                self.evaluate()

    def compute_rewards(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Compute rewards for sequences using reward model.
        
        Args:
            sequences: Generated sequences tensor
            
        Returns:
            Tensor of rewards
        """
        batch_size = sequences.size(0)
        rewards = torch.zeros(batch_size, device=sequences.device)
        
        # Process in smaller batches to avoid OOM
        batch_size = min(32, batch_size)  # Adjust based on available memory
        for i in range(0, sequences.size(0), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            # Get reward model outputs
            with torch.no_grad():
                outputs = self.reward_model(batch_sequences)
                if self.config.use_process_supervision:
                    # Get per-step rewards
                    step_rewards = outputs.logits  # Assuming reward model outputs per-step scores
                else:
                    # Get final reward only
                    step_rewards = outputs.logits[:, -1].unsqueeze(-1)
                
            rewards[i:i+batch_size] = step_rewards
            
        return rewards

    def log_metrics(self, stats: Dict[str, torch.Tensor], update: int, start_time: float):
        """Log training metrics."""
        if not self.is_world_process_zero():
            return
            
        # Calculate average metrics
        metrics = {}
        eps = int(self.state.episode / (time.time() - start_time))
        metrics["eps"] = eps
        
        for key, value in stats.items():
            metrics[f"train/{key}"] = self.accelerator.gather(value).mean().item()
            
        # Add learning rate
        metrics["train/learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        
        # Add episode count
        metrics["train/episodes"] = self.state.episode
        
        # Log metrics
        self.state.log_history.append(metrics)
        if self.args.report_to == ["wandb"]:
            import wandb
            wandb.log(metrics, step=update)
        
        # Print metrics
        self.print_metrics(metrics)

    def print_metrics(self, metrics: Dict[str, float]):
        """Print current training metrics."""
        log_str = f"Update {self.state.episode} | "
        log_str += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if k != "train/episodes"])
        print(log_str)

    def should_save(self, update: int) -> bool:
        """Determine if model should be saved at current update."""
        if not self.args.should_save:
            return False
            
        return (update + 1) % self.args.save_steps == 0

    def should_evaluate(self, update: int) -> bool:
        """Determine if evaluation should be run at current update."""
        if not self.args.do_eval:
            return False
            
        return (update + 1) % self.args.eval_steps == 0

    def save_model(self):
        """Save model checkpoint."""
        if not self.is_world_process_zero():
            return
            
        # Save model
        unwrapped_model = self.accelerator.unwrap_model(self.policy)
        unwrapped_model.save_pretrained(
            self.args.output_dir,
            is_main_process=self.is_world_process_zero(),
            save_function=self.accelerator.save
        )
        
        # Save tokenizer
        if self.is_world_process_zero():
            self.tokenizer.save_pretrained(self.args.output_dir)
            
        # Save training state
        self.save_state()

    def evaluate(self):
        """Run evaluation loop."""
        if not hasattr(self, "eval_dataloader"):
            return
            
        self.policy.eval()
        eval_stats = defaultdict(list)
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                # Generate sequences
                queries = batch["input_ids"].to(self.accelerator.device)
                sequences, _, _ = self.sample_sequences(queries)
                
                # Get rewards
                rewards = self.compute_rewards(sequences)
                
                # Store stats
                eval_stats["rewards"].append(rewards.mean().item())
                
        # Log eval metrics
        metrics = {
            f"eval/{k}": np.mean(v) for k, v in eval_stats.items()
        }
        
        self.state.log_history.append(metrics)
        if self.args.report_to == ["wandb"]:
            import wandb
            wandb.log(metrics, step=self.state.episode)
            
        self.policy.train()

    @classmethod 
    def from_pretrained(
        cls,
        config: GRPOConfig,
        pretrained_model_name_or_path: str,
        reward_model_name_or_path: str = None,
        **kwargs
    ) -> "GRPOTrainer":
        """
        Create a GRPO trainer from pretrained models.

        Args:
            config: GRPO configuration
            pretrained_model_name_or_path: Path or name of pretrained policy model
            reward_model_name_or_path: Path or name of reward model
            **kwargs: Additional arguments for model loading
            
        Returns:
            Configured GRPOTrainer instance
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load models
        policy = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        ref_policy = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        reward_model_path = reward_model_name_or_path or pretrained_model_name_or_path
        reward_model = AutoModelForCausalLM.from_pretrained(reward_model_path, **kwargs)
        
        return cls(
            config=config,
            tokenizer=tokenizer,
            policy=policy,
            ref_policy=ref_policy,
            reward_model=reward_model,
            **kwargs
        )

def main():
    """Example usage of GRPO training."""
    from datasets import load_dataset
    import wandb
    
    # Configuration
    config = GRPOConfig(
        exp_name="math_improvement",
        reward_model_path="reward_model_path",
        num_grpo_epochs=4,
        sampling_group_size=8,
        sampling_strategy="top_p",
        sampling_temperature=0.7,
        # learning_rate=1e-5,
        # num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        output_dir="./grpo_math_model",
        report_to=["wandb"]
    )
    
    # Initialize wandb
    wandb.init(
        project="grpo_math",
        name=config.exp_name,
        config=vars(config)
    )
    
    # Load dataset
    dataset = load_dataset("math_dataset")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    
    # Create trainer
    trainer = GRPOTrainer.from_pretrained(
        config=config,
        pretrained_model_name_or_path="math_base_model",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model()
    
    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main()

# End of implementation