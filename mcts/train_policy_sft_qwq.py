from unsloth import FastLanguageModel
import torch
import wandb
from datasets import load_dataset
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from unsloth.chat_templates import get_chat_template

# Constants
max_seq_length = 32768  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False  # Use 4bit quantization to reduce memory usage. Can be False.

def extract_boxed_solution(text):
    """
    Extract solution from \boxed{} notation and clean it.
    Returns None if no boxed solution is found.
    """
    import re
    try:
        # Find content inside \boxed{}
        match = re.search(r'\\boxed{([^}]+)}', text)
        if match:
            # Extract and clean the solution
            solution = match.group(1)
            # Remove spaces, newlines and extra whitespace
            solution = re.sub(r'\s+', '', solution)
            return solution
        return None
    except Exception as e:
        print(f"Error extracting boxed solution: {str(e)}")
        return None

def filter_dataset(example):
    """
    Filter dataset based on solution matching between boxed answer and QwQ response.
    Returns True if example should be kept, False if it should be filtered out.
    
    Args:
        example: Dictionary containing dataset fields (problem, solution, qwq)
        
    Returns:
        bool: True if example meets criteria, False otherwise
    """
    try:
        # Extract solution from the solution column
        if 'solution' not in example:
            return False
            
        boxed_solution = extract_boxed_solution(example['solution'])
        if not boxed_solution:
            return False
            
        # Clean the QwQ response for comparison
        qwq_clean = ''.join(example['qwq'].split())
        
        # Check if the boxed solution appears in the QwQ response
        if boxed_solution not in qwq_clean:
            return False
            
        # Additional basic quality checks
        if len(example['qwq']) < 50:  # Minimum response length
            return False
        
        # Additional basic quality checks
        if len(example['qwq']) > 10000:  # Max response length
            return False
            
        return True
        
    except Exception as e:
        print(f"Error in filter validation: {str(e)}")
        return False

def train_sft():
    # Load base and instruct models
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-0.5B",
        # model_name = "unsloth/Qwen2.5-Math-1.5B",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    model_instruct, tokenizer_instruct = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-0.5B-Instruct",
        # model_name = "unsloth/Qwen2.5-Math-1.5B-Instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     model_name = "unsloth/Qwen2.5-0.5B-Instruct",
    #     # model_name = "unsloth/Qwen2.5-Math-1.5B-Instruct",
    #     max_seq_length = max_seq_length,
    #     dtype = dtype,
    #     load_in_4bit = load_in_4bit,
    # )

    # TRANSFER = False
    TRANSFER = True
    if TRANSFER:
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
    dataset = load_dataset("qingy2024/QwQ-LongCoT-Verified-130K", "verified", split="train")

    # Apply filtering
    print("filtering")
    filtered_dataset = dataset.filter(filter_dataset)
    print(f"Original dataset size: {len(dataset)}")
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    
    # Print some examples of filtered data
    print("\nExample of filtered data:")
    for idx in range(min(3, len(filtered_dataset))):
        print(f"\nExample {idx + 1}:")
        print("Problem:", filtered_dataset[idx]['problem'][:200], "...")
        print("Response:", filtered_dataset[idx]['qwq'][:200], "...")


    def formatting_prompts_func(examples):
        conversations = []
        for query, response in zip(examples['problem'], examples['qwq']):
            # break
            conversation = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response}
            ]
            conversations.append(conversation)

        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
                for convo in conversations]
        return {"text": texts}

    dataset = filtered_dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)
    print(len(dataset))
    
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
            # learning_rate = 5e-5,
            # embedding_learning_rate = 5e-6,
            # learning_rate = 3e-5,
            # embedding_learning_rate = 3e-6,
            learning_rate = 3e-6,
            embedding_learning_rate = 3e-7,
            # per_device_train_batch_size = 8,  # With gradient_accumulation_steps=8 this gives effective batch size 64
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 8,
            lr_scheduler_type = "cosine",
            num_train_epochs = 3,
            # num_train_epochs = 2,
            # num_train_epochs = 1,
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
            # run_name = "qwqdistill1.5",
            run_name = "qwqdistill",
            hub_strategy = "every_save",
            save_strategy = "steps",
            save_steps = 100,
            hub_model_id = "rawsh/q1-Qwen2.5-0.5b"
            # hub_model_id = "rawsh/q1-Qwen2.5-0.5b-Instruct"
            # hub_model_id = "rawsh/q1-Qwen2.5-Math-1.5B"
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
        "rawsh/q1-Qwen2.5-0.5b",  # Replace with your username
        # "rawsh/q1-Qwen2.5-0.5b-Instruct", 
        # "rawsh/q1-Qwen2.5-Math-1.5B",
        tokenizer, 
        save_method="merged_16bit",
    )

if __name__ == "__main__":
    train_sft()