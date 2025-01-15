import modal

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    # modal.Image.debian_slim()
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
        .apt_install("git")
        .pip_install("torch")
        .pip_install("packaging")
        .pip_install("wheel")
        .run_commands("pip install flash-attn --no-build-isolation")
        .pip_install("transformers")
        .pip_install("accelerate")
        .pip_install("numpy")
        .pip_install("datasets")
        .pip_install("wandb")
        .pip_install("bitsandbytes")
        .pip_install("unsloth @ git+https://github.com/unslothai/unsloth.git")
        .pip_install("unsloth_zoo")
        .pip_install("xformers")
        .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
app = modal.App("train_policy_sft", image=image)

with image.imports():
    from mcts.train_policy_sft_prime import train_sft

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

@app.function(
    cpu=2.0,
    # gpu=modal.gpu.A10G(),
    gpu=modal.gpu.H100(),
    # gpu=modal.gpu.A100(size="40GB"),
    timeout=20 * HOURS,
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("wandb-token")
    ]
)
def train_policy_model_sft_upload_to_hf():
    train_sft()

@app.local_entrypoint()
def main():
    # run the function remotely on Modal
    train_policy_model_sft_upload_to_hf.remote()