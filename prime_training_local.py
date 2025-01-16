# -
# Prereq
# -
# pip3 install torch==2.4.0 packaging wheel
# pip3 install flash-attn
# pip3 install vllm==0.6.3 ray transformers accelerate numpy datasets wandb bitsandbytes tensorboard tqdm evaluate pyext pylatexenc
# export WANDB_API_KEY=""

num_gpus = 8
root_dir = "/home/ubuntu"
model_save_dir = f"{root_dir}/save_dir"
prime_dir = f"{root_dir}/prime/training"
tmp_dir = f"{root_dir}/tmp"
hf_model = "rawsh/SmallThinker-3B"
hf_dataset = "rawsh/Eurus-2-RL-Data-ProblemsOnly"

# sampling
response_length = 9000

def download_and_setup_prime():
    """Download PRIME repository and install dependencies during image build"""
    import os
    import subprocess
    from datasets import load_dataset
    
    # Clone PRIME repository
    subprocess.run(["git", "clone", "https://github.com/PRIME-RL/PRIME.git", "/home/ubuntu/prime"], check=False)
    os.chdir(prime_dir)
    
    # Create data directory
    os.makedirs("data", exist_ok=True)

    # Create tmp dir
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Download dataset from Hugging Face
    print("Downloading dataset from Hugging Face...")
    dataset = load_dataset(hf_dataset)
    
    # Save train and validation splits to parquet files
    print("Saving dataset splits to parquet files...")
    dataset['train'].to_parquet("data/train.parquet")
    dataset['validation'].to_parquet("data/validation.parquet")
    
    # Install training-specific requirements
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True, cwd=prime_dir)
    
    # Add PRIME directories to Python path
    with open("/home/ubuntu/.profile", "a") as f:
        f.write("""
export PYTHONPATH="${PYTHONPATH}:/home/ubuntu/prime/training"
export PYTHONPATH="${PYTHONPATH}:/home/ubuntu/prime"
""")

    # Create modified training script with proper paths
    with open("/home/ubuntu/prime/training/run_train.sh", "w") as f:
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
/home/ubuntu/.local/bin/ray start --head \
    --port=$port \
    --num-gpus={num_gpus} \
    --include-dashboard=false \
    --temp-dir={tmp_dir} \
    --block &

cd /home/ubuntu/prime/training
python3 -m verl.trainer.main_ppo \\
    data.train_files=[/home/ubuntu/prime/training/data/train.parquet] \\
    data.val_files=[/home/ubuntu/prime/training/data/validation.parquet] \\
    data.train_batch_size=256 \\
    data.val_batch_size=1024 \\
    data.max_prompt_length=1024 \\
    data.max_response_length={response_length} \\
    actor_rollout_ref.model.path={hf_model} \\
    actor_rollout_ref.actor.optim.lr=5e-7 \\
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \\
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \\
    actor_rollout_ref.actor.fsdp_config.param_offload=True \\
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \\
    actor_rollout_ref.actor.entropy_coeff=0. \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \\
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    algorithm.kl_ctrl.kl_coef=0.00 \\
    trainer.logger=['console','wandb'] \\
    trainer.project_name=$PROJECT_NAME \\
    trainer.experiment_name=$EXPERIMENT_NAME \\
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \\
    trainer.n_gpus_per_node={num_gpus} \\
    trainer.nnodes=1 \\
    trainer.save_freq=8 \\
    trainer.test_freq=8 \\
    trainer.total_epochs=1 \\
    +trainer.total_training_steps=300 \\
    +trainer.val_before_train=True \\
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
    reward_model.prime_model.path={hf_model} \\
    reward_model.prime_model.ref_path={hf_model} \\
    reward_model.model.input_tokenizer=null \\
    reward_model.micro_batch_size=8 \\
    reward_model.prime_model.ref_type=freeze \\
    reward_model.prime_model.update=after \\
    reward_model.prime_model.beta_train=0.05 \\
    reward_model.prime_model.optim.lr=1e-6 \\
    reward_model.prime_model.optim.grad_clip=10.0 \\
    reward_model.prime_model.input_tokenizer=null""")
    # Make executable
    os.chmod("/home/ubuntu/prime/training/run_train.sh", 0o755)


def train_prime():
    """Main training function for PRIME"""
    import os
    import subprocess
    
    print("Starting PRIME training...")
    print("\nCurrent working directory:", os.getcwd())
    
    os.chdir("/home/ubuntu/prime/training")
    print("PYTHONPATH:", os.environ.get('PYTHONPATH'))
    print("\nDirectory contents:")
    subprocess.run(["ls", "-la"], check=True)
    subprocess.run(["/home/ubuntu/prime/training/run_train.sh"], check=True)

if __name__ == "__main__":
    download_and_setup_prime()
    train_prime()