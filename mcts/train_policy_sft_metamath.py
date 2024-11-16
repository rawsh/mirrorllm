from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

# Constants
SEED = 42
max_seq_length = 8192
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False

first = True

def format_answer(response):
    global first
    """Extract answer from #### pattern and format response."""
    # Split at #### and get everything before it
    parts = response.split('####')
    if len(parts) < 2:
        return None
    
        
    solution = "\n\n".join(parts[0].strip().split("\n"))
    answer = parts[1].split('The answer is:')[0].strip()

    if (first):
        print(solution)
        print(answer)
        first = False
    
    return f"{solution}\n\nThe final answer is: $\\boxed{{{answer}}}$"

def train_sft():
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Qwen/Qwen2.5-0.5B",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # Set up chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5",
    )

    # Configure PEFT
    model = FastLanguageModel.get_peft_model(
        model,
        r = 128,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = True,
        loftq_config = None,
    )

    # Load dataset
    ds = load_dataset("meta-math/MetaMathQA")
    train_ds = ds['train']

    # Format prompts
    def formatting_prompts_func(examples):
        conversations = []
        for query, response in zip(examples['query'], examples['response']):
            formatted_response = format_answer(response)
            if formatted_response is None:
                continue

            conversation = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": formatted_response}
            ]
            conversations.append(conversation)
        
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
                for convo in conversations]
        return {"text": texts}
    
    # <|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the total cost of purchasing equipment for all sixteen players on the football team, considering that each player requires a $25 jersey, a $15.20 pair of shorts, and a pair of socks priced at $6.80?<|im_end|>\n<|im_start|>assistant\nEach player requires a $25 jersey, a $15.20 pair of shorts, and a pair of socks priced at $6.80.\n\nSo the total cost for each player is $25 + $15.20 + $6.80 = $47.\n\nSince there are sixteen players on the football team, the total cost for all of them is 16 * $47 = $752.\n\nThe final answer is: $\\boxed{752}$<|im_end|>\n'

    # Process dataset
    formatted_dataset = train_ds.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=train_ds.column_names
    )
    print(len(formatted_dataset))
    print(formatted_dataset[0])

    # Configure trainer
    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = formatted_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 8,
        packing = True,
        args = UnslothTrainingArguments(
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 8,
            warmup_ratio = 0.1,
            num_train_epochs = 3,
            # learning_rate = 5e-6,
            # embedding_learning_rate = 5e-7,
            learning_rate = 8e-6,
            embedding_learning_rate = 1e-6,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_torch_fused",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    # Print GPU stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Train
    trainer_stats = trainer.train()

    # Save model
    model.push_to_hub_merged(
        "rawsh/MetaMath-Qwen2.5-0.5b", 
        tokenizer, 
        save_method = "merged_16bit"
    )

if __name__ == "__main__":
    train_sft()