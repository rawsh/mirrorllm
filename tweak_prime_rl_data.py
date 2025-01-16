from datasets import load_dataset
import json
from huggingface_hub import login

dataset = load_dataset("rawsh/Eurus-2-RL-Data-ProblemsOnly", split="validation")
sample = dataset[100]
print("\nSample processed chat trace:")
print(sample['prompt'][0]["content"])
import sys
sys.exit()

# Load the dataset
dataset = load_dataset("PRIME-RL/Eurus-2-RL-Data")

def process_chat_trace(example):
    """Process a single example by removing the system prompt from the chat trace."""
    try:
        # Parse the chat trace from string to list
        chat_trace = example['prompt']
        
        # Remove the first message (system prompt)
        modified_trace = chat_trace[1:]
        
        # Convert back to string
        example['prompt'] = modified_trace
        
        return example
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"Error processing example: {e}")
        return example

# Process each split in the dataset
processed_dataset = {}
for split in dataset:
    processed_dataset[split] = dataset[split].map(process_chat_trace)

# You'll need to run this first and enter your token when prompted
login()

# Push to HF Hub
for split, data in processed_dataset.items():
    data.push_to_hub(
        "rawsh/Eurus-2-RL-Data-ProblemsOnly",
        split=split,
        private=False  # Set to True if you want a private repository
    )

# Print sample to verify
sample = processed_dataset['train'][0] if 'train' in processed_dataset else next(iter(processed_dataset.values()))[0]
print("\nSample processed chat trace:")
print(sample['prompt'])