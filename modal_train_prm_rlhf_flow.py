# train.py
import modal
import yaml
import os
from pathlib import Path

# CUDA setup
AXOLOTL_REGISTRY_SHA = "9578c47333bdcc9ad7318e54506b9adaf283161092ae780353d506f7a656590a"
image = (
    modal.Image.from_registry(f"winglian/axolotl@sha256:{AXOLOTL_REGISTRY_SHA}")
    .pip_install(
        "huggingface_hub==0.23.2",
        "hf-transfer==0.1.5",
        "wandb==0.16.3",
        "fastapi==0.110.0",
        "pydantic==2.6.3",
    )
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            AXOLOTL_NCCL_TIMEOUT="60",
        )
    )
    .entrypoint([])
)

app = modal.App("train-hf", image=image)

# Constants
MINUTES = 60
HOURS = 60 * MINUTES

# Create volume for persistent storage
training_vol = modal.Volume.from_name("training-data", create_if_missing=True)

@app.function(
    cpu=8,
    gpu=modal.gpu.H100(),
    timeout=20 * HOURS,
    volumes={"/training": training_vol},
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("wandb-token")
    ],
)
def run_training(config):
    import subprocess
    
    # Write the config to the container
    config_path = Path("/training/config.yml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # Run training - Axolotl will handle HF upload if push_to_hub is True
    subprocess.run(["python", "-m", "axolotl.cli.train", config_path])

@app.local_entrypoint()
def main():
    # Read the local config file
    with open("prm_rlhf_flow/qwen.yml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Run the training
    run_training.remote(config)

if __name__ == "__main__":
    main()