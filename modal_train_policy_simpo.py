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
    from mcts.train_policy_simpo import train_simpo  # Import from our new simplified script

# Create Modal app
app = modal.App("train-policy-simpo", image=image)

@app.function(
    cpu=4.0,
    gpu=modal.gpu.H100(count=1),
    timeout=24 * 60 * 60,
    memory=32768,
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("wandb-token")
    ],
)
def train_policy_simpo():
    import os
    from huggingface_hub import HfFolder
    import wandb
    
    try:
        # Set up HuggingFace token
        hf_token = os.environ["HF_TOKEN"]
        HfFolder.save_token(hf_token)
        
        # Set up Weights & Biases
        wandb.login(key=os.environ["WANDB_API_KEY"])
        
        # Run training with specified parameters
        train_simpo(
            # model_name="rawsh/mirrorqwen2.5-0.5b-SFT",
            # model_name="rawsh/mirrorqwen2.5-0.5b-SimPO-0",
            # model_name="rawsh/mirrorqwen2.5-0.5b-SimPO-1",
            model_name="rawsh/mirrorqwen2.5-0.5b-SimPO-2",
            dataset_name="rawsh/mirrorqwen2.5-0.5B-gsm8k-policy-data-ST-3",
            output_model_name="rawsh/mirrorqwen2.5-0.5b-SimPO-3",
            hub_token=hf_token
        )
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
    print("Starting full model SimPO training on Modal...")
    try:
        train_policy_simpo.remote()
        print("Training job submitted to Modal. Check W&B dashboard for training progress.")
    except Exception as e:
        print(f"Error in training job: {str(e)}")
        sys.exit(1)