from unsloth import FastLanguageModel
import torch

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

from datasets import load_dataset


# DUPLICATED CODE FOR MODAL
# ---------------------
import re
SEED = 42

def split_and_clean_steps(text):
    # Use regex to split the text into steps
    steps = re.split(r'(?=##\s*Step\s+\d+:)', text)
    
    # Remove any leading/trailing whitespace, empty steps, and the "## Step n:" prefix
    cleaned_steps = []
    for step in steps:
        # Strip whitespace and check if step is not empty
        step = step.strip()
        if step:
            # Remove the "## Step n:" prefix
            step = re.sub(r'^##\s*Step\s+\d+:\s*', '', step)
            cleaned_steps.append(step)
    
    return cleaned_steps

def quality_filter(example):
    response_quality = example['score'] >= 0.32 # arbitrary af 
    # TODO: check correctness of chain
    # math_and_reasoning = example['primary_tag'] in ['Math', 'Reasoning']
    instruction_quality = example['quality'] in ['excellent', 'good']
    response_format = "## Step 1: " in example['response']
    return response_quality and instruction_quality and response_format
# ---------------------


def train_sft():
    max_seq_length = 8192 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/gemma-2-2b",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head",], # Add for continual pretraining
        lora_alpha = 32,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = True,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )


    # dataset
    ds = load_dataset("argilla/magpie-ultra-v0.1")
    filtered_ds = ds.filter(quality_filter)
    split_ds = filtered_ds['train'].train_test_split(test_size=0.1, seed=SEED)
    train_ds = split_ds['train']

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        texts = []
        for instruction, response in zip(examples['instruction'], examples['response']):
            clean_steps = split_and_clean_steps(response)
            all_steps = "\n\n".join(clean_steps)

            prompt = f"{instruction}\n\n{all_steps}{EOS_TOKEN}"
            texts.append(prompt)
        
        return {"text": texts}
    formatted_dataset = train_ds.map(formatting_prompts_func, batched = True,)


    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = formatted_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 8,
        packing = True,

        args = UnslothTrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 8,

            warmup_ratio = 0.1,
            num_train_epochs = 1,

            learning_rate = 4e-4,
            embedding_learning_rate = 4e-5,

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

    #@title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    model.push_to_hub_merged("rawsh/mirrorgemma-2-2b-SFT", tokenizer, save_method = "merged_16bit")