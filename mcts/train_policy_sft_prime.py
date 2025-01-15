from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from unsloth.chat_templates import get_chat_template
from typing import List, Dict
import json

# Constants
max_seq_length = 8192
dtype = None
load_in_4bit = False
BATCH_SIZE = 10000  # Batch size for tokenization

def process_conversation_batch(examples: Dict, tokenizer) -> List[str]:
    """Process a batch of conversations and return formatted chat templates."""
    conversations = []
    
    for system, conv_list in zip(examples['system'], examples['conversations']):
        try:
            # Basic validation
            if not conv_list or len(conv_list) < 2:
                continue
            if not (conv_list[0].get('from') == 'human' and conv_list[1].get('from') == 'gpt'):
                continue

            # Format messages
            formatted_msgs = [{"role": "system", "content": system}]
            formatted_msgs.extend([
                {"role": "user" if msg['from'] == 'human' else "assistant", "content": msg['value']}
                for msg in conv_list
            ])
            conversations.append(formatted_msgs)
            
        except (json.JSONDecodeError, AttributeError, KeyError):
            continue
    
    # Apply chat template without tokenization
    return [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) 
            for conv in conversations]

def filter_by_length(texts: List[str], tokenizer, max_length: int) -> List[str]:
    """Filter texts by tokenized length."""
    tokenized = tokenizer(texts, truncation=False, padding=False)
    return [text for i, text in enumerate(texts) 
            if len(tokenized["input_ids"][i]) < max_length]

def train_sft():
    # Load base and instruct models
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-0.5B",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    model_instruct, tokenizer_instruct = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-0.5B-Instruct",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Transfer embeddings
    TRANSFER = True
    if TRANSFER:
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

    # Cleanup
    import gc
    del model_instruct, tokenizer_instruct
    gc.collect()
    torch.cuda.empty_cache()

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=128,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
        loftq_config=None,
    )

    # Setup tokenizer
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    tokenizer.eos_token = "<|im_end|>"

    # Load dataset
    dataset = load_dataset("PRIME-RL/Eurus-2-SFT-Data", split="train")

    def formatting_prompts_func(examples):
        # Process conversations in the current batch
        texts = process_conversation_batch(examples, tokenizer)
        
        # Filter by tokenized length
        texts_filtered = filter_by_length(texts, tokenizer, max_seq_length)
        
        if len(texts) != len(texts_filtered):
            print(f"Filtered {len(texts) - len(texts_filtered)} examples due to length")
            
        return {"text": texts_filtered}

    # Process dataset
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=dataset.column_names
    )
    
    print(f"Final dataset size: {len(dataset)}")

    # Configure trainer
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=64,
        args=UnslothTrainingArguments(
            learning_rate=5e-6,
            embedding_learning_rate=5e-7,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=8,
            lr_scheduler_type="cosine",
            num_train_epochs=3,
            warmup_ratio=0.1,
            max_seq_length=max_seq_length,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            optim="adamw_8bit",
            weight_decay=0.01,
            logging_steps=1,
            seed=3407,
            output_dir="outputs",
            report_to="wandb",
            run_name="eurus-sft",
            hub_strategy="every_save",
            save_strategy="steps",
            save_steps=100,
            hub_model_id="rawsh/Qwen2.5-0.5b-Eurus-2-SFT"
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

    # Show memory stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory/max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)

    print(f"Training time: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
    print(f"Peak memory usage: {used_memory} GB ({used_percentage}%)")
    print(f"LoRA training memory: {used_memory_for_lora} GB ({lora_percentage}%)")

    # Save to HuggingFace Hub
    model.push_to_hub_merged(
        "rawsh/Qwen2.5-0.5b-Eurus-2-SFT",
        tokenizer,
        save_method="merged_16bit",
    )

if __name__ == "__main__":
    train_sft()