from dataclasses import dataclass, field
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import CPOConfig, CPOTrainer
import wandb

@dataclass
class ScriptArguments:
    model_name: str = field(default="Qwen/Qwen2-0.5B-Instruct")
    dataset_name: str = field(default="rawsh/mirrorqwen2.5-0.5B-gsm8k-policy-data-ST-0")
    output_dir: str = field(default="simpo-math-model")
    warmup_ratio: float = field(default=0.1)  # 10% warmup
    lr_scheduler_type: str = field(default="cosine")  # Cosine decay
    max_grad_norm: float = field(default=1.0)
    output_model_name: str = field(default=None)
    hub_token: str = field(default=None)
    push_to_hub: bool = field(default=True)
    # learning_rate: float = field(default=3e-7)
    learning_rate: float = field(default=5e-7)
    batch_size: int = field(default=8)
    num_train_epochs: int = field(default=7)
    # max_steps: int = field(default=-1)
    # max_steps: int = field(default=10)
    gradient_accumulation_steps: int = field(default=8)
    beta: float = field(default=2.0)
    simpo_gamma: float = field(default=0.5)

# class CustomCPOTrainer(CPOTrainer):
#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
#         wandb.log({"loss": loss.item()}, step=self.state.step)
#         if return_outputs:
#             return loss, outputs
#         return loss

def train_simpo(
        model_name=None,
        dataset_name=None,
        output_model_name=None, 
        hub_token=None
    ):
    args = ScriptArguments()
    if model_name:
        args.model_name = model_name
    if dataset_name:
        args.dataset_name = dataset_name
    if output_model_name:
        args.output_model_name = output_model_name
    if hub_token:
        args.hub_token = hub_token

    wandb.init(project="simpo-training")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.config.use_cache = False

    dataset = load_dataset(args.dataset_name, token=args.hub_token)
    train_dataset = dataset["train"].map(
        lambda examples: {
            "prompt": examples["question"],
            "chosen": ["\n\n".join(ex["steps"]) for ex in examples["positive"]],
            "rejected": ["\n\n".join(ex["steps"]) for ex in examples["negative"]]
        },
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    training_args = CPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        # max_steps=args.max_steps,
        remove_unused_columns=False,
        loss_type="simpo",
        cpo_alpha=0.5,
        beta=args.beta,
        simpo_gamma=args.simpo_gamma,
        max_length=2048,
        max_prompt_length=1024,
        gradient_checkpointing=True,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.output_model_name,
        hub_token=args.hub_token,
        hub_strategy="end",
        report_to=["wandb"],
        # Mixed precision settings
        bf16=True,  # Use bfloat16 instead of fp16
        tf32=True,
        optim="paged_adamw_32bit",  # Use 32-bit optimizer
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=20,
    )

    trainer = CPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        processing_class=tokenizer
    )

    trainer.train()
    trainer.save_model()

    # if args.push_to_hub and args.output_model_name:
    #     print("saving model")
    #     trainer.push_to_hub(repo_id=args.output_model_name, commit_message="Final SimPO model")
    #     tokenizer.push_to_hub(repo_id=args.output_model_name)
    
    wandb.finish()

if __name__ == "__main__":
    train_simpo()