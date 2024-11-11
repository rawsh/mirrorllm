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
        .pip_install("matplotlib")
        .pip_install("seaborn")
)
app = modal.App("train_prm", image=image)

with image.imports():
    from mcts.train_reward import train_reward_model

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

vol = modal.Volume.from_name("prm-tmp", create_if_missing=True)

@app.function(
    cpu=2.0,
    # gpu=modal.gpu.A10G(),
    gpu=modal.gpu.H100(),
    # gpu=modal.gpu.A100(count=4, size="40GB"),
    # gpu=modal.gpu.A100(size="40GB"),
    timeout=20 * HOURS,
    secrets=[
        modal.Secret.from_name("hf-token"),
        modal.Secret.from_name("wandb-token")
    ],
    volumes={"/out": vol},
)
def train_reward_model_upload_to_hf():
    train_reward_model(
        # add revision
        model_name="Qwen/Qwen2.5-0.5B",
        dataset_path="rawsh/magpie-ultra-v0.1-PRM-data-base",
        output_model_name="rawsh/mirrorqwen2.5-0.5b-prm",
        disable_binning=False
    )

@app.local_entrypoint()
def main():
    # run the function remotely on Modal
    train_reward_model_upload_to_hf.remote()