from unsloth import FastLanguageModel
import torch
import wandb
from datasets import load_dataset
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from unsloth.chat_templates import get_chat_template

# Constants
max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.

first_type1 = True
first_type2 = True

def format_answer(response):
    """Extract answer from #### pattern and format response."""
    global first_type1
    global first_type2

    # Split at #### and get everything before it
    parts = response.split('####')
    if len(parts) < 2:
        # combine the last two steps
        steps = parts[0].strip().split("\n")
        if len(steps) > 1:
            steps[-2] = steps[-2] + f"\n{steps[-1]}"
            steps = steps[:-1]
            sol = "\n\n".join(steps)

            if (first_type1):
                print(response)
                first_type1 = False

            return sol
        else:
            return None

    solution = "\n\n".join(parts[0].strip().split("\n"))
    answer = parts[1].split('The answer is:')
    answer = answer[0].strip()
    sol = f"{solution}\nThe answer is: {answer}"

    if (first_type2):
        print(response)
        first_type2 = False

    return sol

def train_sft():
    # Load base and instruct models
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-0.5B",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    model_instruct, tokenizer_instruct = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-0.5B-Instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # Transfer chat token embeddings from instruct to base model
    base_embeddings = model.get_input_embeddings()
    instruct_embeddings = model_instruct.get_input_embeddings()
    chat_tokens = ["<|im_start|>", "<|im_end|>", "system", "assistant", "user"]
    with torch.no_grad():
        for token in chat_tokens:
            try:
                instruct_id = tokenizer_instruct.convert_tokens_to_ids(token)
                base_id = tokenizer.convert_tokens_to_ids(token)
                if instruct_id != tokenizer_instruct.unk_token_id and base_id != tokenizer.unk_token_id:
                    base_embeddings.weight[base_id] = instruct_embeddings.weight[instruct_id].clone()
                    print(f"Transferred embedding for token: {token}")
                else:
                    print(f"Warning: Token {token} not found in one of the vocabularies")
            except Exception as e:
                print(f"Error transferring token {token}: {str(e)}")

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 128,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head",],  # Add for continual pretraining
        lora_alpha = 32,
        lora_dropout = 0,  # Supports any, but = 0 is optimized
        bias = "none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth",  # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = True,  # We support rank stabilized LoRA
        loftq_config = None,  # And LoftQ
    )

    # Set up tokenizer with chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5",
    )
    tokenizer.eos_token = "<|im_end|>"
    print(tokenizer.eos_token)
    print(tokenizer.pad_token)

    # Load and process dataset
    dataset = load_dataset("meta-math/MetaMathQA", split="train")

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

    dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)
    
    # Debug tokenizer output - show examples
    print("Example of tokenized output:")
    print(dataset[5]["text"])
    print("\nAnother example:")
    print(dataset[100]["text"])

    # Configure trainer
    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 8,

        args = UnslothTrainingArguments(
            learning_rate = 5e-5,
            embedding_learning_rate = 5e-6,
            per_device_train_batch_size = 8,  # With gradient_accumulation_steps=8 this gives effective batch size 64
            gradient_accumulation_steps = 8,
            lr_scheduler_type = "cosine",
            num_train_epochs = 3,
            warmup_ratio = 0.1,
            max_seq_length = 2048,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            optim = "adamw_8bit",
            weight_decay = 0.01,
            logging_steps = 1,
            seed = 3407,
            output_dir = "outputs",
            report_to = "wandb",
            run_name = "metamath",
            hub_strategy = "every_save",
            save_strategy = "steps",
            save_steps = 100,
            hub_model_id = "rawsh/MetaMath-Qwen2.5-0.5b"
        ),
    )

    # Set up wandb
    # wandb.login(key="YOUR_WANDB_KEY")  # Replace with your key
    # wandb.init(project='metamath')

    # Print initial GPU stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Train
    trainer_stats = trainer.train()

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory/max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Save model to HuggingFace Hub
    model.push_to_hub_merged(
        "rawsh/MetaMath-Qwen2.5-0.5b",  # Replace with your username
        tokenizer, 
        save_method="merged_16bit",
    )

if __name__ == "__main__":
    train_sft()