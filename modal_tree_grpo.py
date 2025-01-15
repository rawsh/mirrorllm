import modal
import sys
import traceback

# Define CUDA specifications
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Create Modal image with all necessary dependencies
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git")
    .pip_install("torch")
    .pip_install("transformers")
    .pip_install("accelerate")
    .pip_install("datasets")
    .pip_install("wandb")
    .pip_install("trl>=0.7.6")
    .pip_install("huggingface_hub")
    .pip_install("bitsandbytes")
)

with image.imports():
    from reinforcement_learning.tree_grpo import GRPOConfig, GRPOTrainer

# Create Modal app
app = modal.App("train-policy-tree-grpo", image=image)

@app.function(
    cpu=4.0,
    # gpu=modal.gpu.H100(count=1),
    gpu=modal.gpu.A10G(),
    timeout=24 * 60 * 60,
    # memory=32768,
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("wandb-token")
    ],
)
def train_policy_grpo():
    import os
    from huggingface_hub import HfFolder
    from datasets import load_dataset
    import wandb
    
    try:
        # Set up HuggingFace token
        hf_token = os.environ["HF_TOKEN"]
        HfFolder.save_token(hf_token)
        
        # Set up Weights & Biases
        wandb.login(key=os.environ["WANDB_API_KEY"])
        
        # Configuration
        config = GRPOConfig(
            exp_name="math_improvement",
            reward_model_path="rawsh/MetaMath-Qwen2.5-0.5b-PRM",
            num_grpo_epochs=4,
            sampling_group_size=8,
            sampling_strategy="top_p",
            sampling_temperature=0.7,
            # learning_rate=1e-5,
            # num_train_epochs=3,
            # per_device_train_batch_size=4,
            # gradient_accumulation_steps=4,
            # output_dir="./grpo_math_model",
            # report_to=["wandb"]
        )
        
        # Initialize wandb
        wandb.init(
            project="grpo_math",
            name=config.exp_name,
            config=vars(config)
        )
        
        # Load dataset
        train_dataset = load_dataset("lighteval/MATH", "all", split="train")
        eval_dataset = load_dataset("lighteval/MATH", "all", split="test")
        
        # Create trainer
        trainer = GRPOTrainer.from_pretrained(
            config=config,
            pretrained_model_name_or_path="rawsh/MetaMath-Qwen2.5-0.5b",
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        # Train
        trainer.train()
        
        # Save final model
        trainer.save_model()
        
        # Close wandb
        wandb.finish()
    except Exception as e:
        print(f"Error during training: {str(e)}", file=sys.stderr)
        print("Traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Make sure to finish wandb run even on error
        try:
            wandb.finish()
        except:
            pass
        raise e

@app.local_entrypoint()
def main():
    print("Starting full model GRPO training on Modal...")
    try:
        train_policy_grpo.remote()
        print("Training job submitted to Modal. Check W&B dashboard for training progress.")
    except Exception as e:
        print(f"Error in training job: {str(e)}")
        sys.exit(1)