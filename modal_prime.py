
# CUDA configuration
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

num_gpus = 1
model_save_dir = "/save_dir"


import modal
# TRAIN_GPU = modal.gpu.A10G()
TRAIN_GPU = modal.gpu.H100()
# TRAIN_GPU = modal.gpu.L40S()
vol = modal.Volume.from_name("prime_savedir", create_if_missing=True)

def download_and_setup_prime():
    """Download PRIME repository and install dependencies during image build"""
    import os
    import subprocess
    from datasets import load_dataset
    
    # Clone PRIME repository
    subprocess.run(["git", "clone", "https://github.com/PRIME-RL/PRIME.git", "/PRIME"], check=True)
    os.chdir("/PRIME/training")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Download dataset from Hugging Face
    print("Downloading dataset from Hugging Face...")
    dataset = load_dataset("PRIME-RL/Eurus-2-RL-Data")
    
    # Save train and validation splits to parquet files
    print("Saving dataset splits to parquet files...")
    dataset['train'].to_parquet("data/train.parquet")
    dataset['validation'].to_parquet("data/validation.parquet")
    
    # Install training-specific requirements
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
    
    # Add PRIME directories to Python path
    with open("/etc/profile.d/prime_paths.sh", "w") as f:
        f.write("""
export PYTHONPATH="${PYTHONPATH}:/PRIME/training"
export PYTHONPATH="${PYTHONPATH}:/PRIME"
""")

    # Create modified training script with proper paths
    with open("/PRIME/training/run_train.sh", "w") as f:
        f.write(f"""#!/bin/bash
set -x

# Environment variables
export NCCL_DEBUG=WARN
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true

# Project configuration
export CKPT_PATH='{model_save_dir}'
export PROJECT_NAME='PRIME'
export EXPERIMENT_NAME='test-run'
port=6379

# Start Ray
ray start --head \
    --port=$port \
    --num-gpus=1 \
    --include-dashboard=false \
    --block &

cd /PRIME/training
python3 -m verl.trainer.main_ppo \\
    data.train_files=[/PRIME/training/data/train.parquet] \\
    data.val_files=[/PRIME/training/data/validation.parquet] \\
    data.train_batch_size=256 \\
    data.val_batch_size=1024 \\
    data.max_prompt_length=1024 \\
    data.max_response_length=3072 \\
    actor_rollout_ref.model.path=rawsh/Qwen2.5-0.5b-Eurus-2-SFT \\
    actor_rollout_ref.actor.optim.lr=5e-7 \\
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \\
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \\
    actor_rollout_ref.actor.fsdp_config.param_offload=False \\
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \\
    actor_rollout_ref.actor.entropy_coeff=0. \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \\
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \\
    actor_rollout_ref.ref.fsdp_config.param_offload=False \\
    algorithm.kl_ctrl.kl_coef=0.00 \\
    trainer.logger=['console','wandb'] \\
    trainer.project_name=$PROJECT_NAME \\
    trainer.experiment_name=$EXPERIMENT_NAME \\
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \\
    trainer.n_gpus_per_node={num_gpus} \\
    trainer.nnodes=1 \\
    trainer.save_freq=32 \\
    trainer.test_freq=32 \\
    trainer.total_epochs=1 \\
    data.n_samples=4 \\
    data.filter_accuracy=True \\
    data.accuracy_lower_bound=0.2 \\
    data.accuracy_upper_bound=0.8 \\
    algorithm.adv_estimator=rloo \\
    algorithm.adv_params.verifier_gamma=1.0 \\
    algorithm.adv_params.reward_model_gamma=1.0 \\
    reward_model.rm_type=prime \\
    reward_model.rm_coef=5 \\
    reward_model.prime_granularity=token \\
    reward_model.prime_norm=batch_norm \\
    reward_model.prime_model.path=rawsh/Qwen2.5-0.5b-Eurus-2-SFT \\
    reward_model.prime_model.ref_path=rawsh/Qwen2.5-0.5b-Eurus-2-SFT \\
    reward_model.model.input_tokenizer=null \\
    reward_model.micro_batch_size=8 \\
    reward_model.prime_model.update=after \\
    reward_model.prime_model.beta_train=0.05 \\
    reward_model.prime_model.optim.lr=1e-6 \\
    reward_model.prime_model.optim.grad_clip=10.0 \\
    reward_model.prime_model.input_tokenizer=null""")
    # Make executable
    os.chmod("/PRIME/training/run_train.sh", 0o755)

prime_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "build-essential", "ninja-build")
    # First install torch and its dependencies
    .pip_install([
        "torch==2.4.0",
        "packaging",
        "wheel",
    ])
    # Then install flash-attention separately
    .pip_install("flash-attn")
    # Finally install the rest of the dependencies
    .pip_install([
        "vllm==0.6.3",
        "ray",
        "transformers",
        "accelerate",
        "numpy",
        "datasets",
        "wandb",
        "bitsandbytes",
        "tensorboard",
        "tqdm",
        "evaluate"
    ])
    .pip_install([
        "pyext",
        "pylatexenc"
    ])
    .run_function(download_and_setup_prime, gpu=TRAIN_GPU)
)

app = modal.App("prime-training", image=prime_image)

@app.function(
    cpu=2.0,
    gpu=TRAIN_GPU,
    volumes={model_save_dir: vol},
    timeout=20 * 60 * 60,  # 20 hours
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("wandb-token")
    ]
)
def train_prime():
    """Main training function for PRIME"""
    import os
    import subprocess
    
    print("Starting PRIME training...")
    print("\nCurrent working directory:", os.getcwd())
    
    os.chdir("/PRIME/training")
    print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
    print("\nDirectory contents:")
    subprocess.run(["ls", "-la"], check=True)
    subprocess.run(["/PRIME/training/run_train.sh"], check=True)

@app.local_entrypoint()
def main():
    train_prime.remote()