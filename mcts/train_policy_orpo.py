from dataclasses import dataclass, field
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import ModelConfig, ORPOConfig, ORPOTrainer, get_peft_config
import wandb

@dataclass
class ScriptArguments:
    model_name: str = field(default="Qwen/Qwen2-0.5B-Instruct")
    dataset_name: str = field(default="rawsh/mirrorqwen2.5-0.5B-gsm8k-policy-data-ST-0")
    dataset_train_split: str = field(default="train")
    dataset_test_split: str = field(default="train")  # Using train as test since original doesn't have test split
    output_model_name: str = field(default=None)
    hub_token: str = field(default=None)
    use_peft: bool = field(default=False)

@dataclass
class ModelArguments(ModelConfig):
    model_name_or_path: str = field(default="Qwen/Qwen2-0.5B-Instruct")
    trust_remote_code: bool = field(default=True)
    
def train_orpo(
        model_name=None,
        dataset_name=None,
        output_model_name=None, 
        hub_token=None
    ):
    # Initialize wandb
    wandb.init(project="orpo-training")
    
    # Initialize base arguments
    script_args = ScriptArguments()
    if model_name:
        script_args.model_name = model_name
    if dataset_name:
        script_args.dataset_name = dataset_name
    if output_model_name:
        script_args.output_model_name = output_model_name
    if hub_token:
        script_args.hub_token = hub_token

    # Set up model arguments
    model_args = ModelArguments(
        model_name_or_path=script_args.model_name
    )

    # Set up training configuration
    training_args = ORPOConfig(
        output_dir="orpo-math-model",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        # learning_rate=5e-7,
        # learning_rate=8e-6,
        lr_scheduler_type="linear",
        beta=0.1,
        learning_rate=3e-6,
        # max_steps
        max_length=2048,
        max_prompt_length=1024,
        gradient_checkpointing=True,
        push_to_hub=True,
        hub_model_id=script_args.output_model_name,
        hub_strategy="end",
        report_to=["wandb"],
        bf16=True,
        tf32=True,
        optim="paged_adamw_32bit",
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        # lr_scheduler_type="cosine",
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=10,
        remove_unused_columns=False,
        logging_steps=10,
        logging_first_step=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.config.use_cache = False

    # Load and process dataset
    dataset = load_dataset(script_args.dataset_name, token=script_args.hub_token)
    train_dataset = dataset["train"].map(
        lambda examples: {
            "prompt": examples["question"],
            "chosen": ["\n\n".join(ex["steps"]) for ex in examples["positive"]],
            "rejected": ["\n\n".join(ex["steps"]) for ex in examples["negative"]]
        },
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Initialize trainer
    trainer = ORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args) if script_args.use_peft else None,
    )

    # Train the model
    trainer.train()
    trainer.save_model()
    
    wandb.finish()

if __name__ == "__main__":
    train_orpo()