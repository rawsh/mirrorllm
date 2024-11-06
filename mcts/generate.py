import asyncio
from openai import AsyncOpenAI
import json
from typing import List, Tuple
from datasets import load_dataset
from util import split_and_clean_steps, quality_filter, SEED
from tqdm import tqdm

client = AsyncOpenAI(
    api_key="9FF74944EED19865193F979942FB1",adfghk
    base_url="https://rawsh--vllm-smollm-serve.modal.run/v1"
)

def format_thoughts(thoughts: List[str]) -> str:
    return "\n".join(f"## Step {i}:\n{thought}" for i, thought in enumerate(thoughts, 1))

# template = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n\
# <|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant_partial}"

template = "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n\
<|im_start|>human\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant_partial}"

class ReasoningTrace:
    def __init__(self, question: str, previous_thoughts: List[str], next_step: int):
        self.question = question
        self.previous_thoughts = previous_thoughts
        self.next_step = next_step

class ProcessedReasoningTrace:
    def __init__(self, question: str, thoughts: List[str]):
        self.question = question
        self.thoughts = thoughts

async def generate_thought_batched(batch: List[ReasoningTrace]) -> List[ProcessedReasoningTrace]:
    prompts = []
    for trace in batch:
        formatted_thoughts = format_thoughts(trace.previous_thoughts)
        prompt = template.format(user=trace.question, assistant_partial=f"{formatted_thoughts}\n## Step {trace.next_step}:\n")
        prompts.append(prompt)

    params = {
        # "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "prompt": prompts,
        "max_tokens": 200,
        # "temperature": 0.7,
        "temperature": 0.0,
        "stop": ["\n## Step"],
        "timeout": 600
    }

    try:
        response = await client.completions.create(**params)
        processed = [
            ProcessedReasoningTrace(
                question=batch[i].question,
                thoughts=batch[i].previous_thoughts + [response.choices[i].text.strip()]
            ) for i in range(len(batch))
        ]
        return processed
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

async def format_thought_chain(question: str, chain: List[str]) -> List[ReasoningTrace]:
    return [ReasoningTrace(question, chain[:i], i+1) for i in range(0, len(chain))]

async def process_batch(batch: List[ReasoningTrace], semaphore: asyncio.Semaphore) -> List[ProcessedReasoningTrace]:
    async with semaphore:
        return await generate_thought_batched(batch)

async def process_all_thought_chains_batched(thought_chains: List[Tuple[str, List[str]]]) -> List[ProcessedReasoningTrace]:
    batch_size = 200
    all_traces = []
    
    for question, chain in thought_chains:
        all_traces.extend(await format_thought_chain(question, chain))
    
    results = []
    semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent batches
    tasks = []

    for i in range(0, len(all_traces), batch_size):
        batch = all_traces[i:i + batch_size]
        task = asyncio.create_task(process_batch(batch, semaphore))
        tasks.append(task)

    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing batches"):
        processed_batch = await task
        if processed_batch:
            results.extend(processed_batch)
    
    return results

async def main():
    ds = load_dataset("argilla/magpie-ultra-v0.1")
    filtered_ds = ds.filter(quality_filter)
    split_ds = filtered_ds['train'].train_test_split(test_size=0.1, seed=SEED)
    train_ds = split_ds['train']
    correct_traces = [(row["instruction"], split_and_clean_steps(row["response"])) for row in train_ds]

    # correct_traces = correct_traces[:1000]
    generated_thoughts = await process_all_thought_chains_batched(correct_traces)
    
    with open("out.jsonl", "w") as f:
        for chain in generated_thoughts:
            json.dump(chain.__dict__, f)
            f.write("\n")

    print(f"Results written to out.jsonl")

if __name__ == "__main__":
    asyncio.run(main())