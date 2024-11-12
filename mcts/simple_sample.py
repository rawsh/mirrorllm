import asyncio
import aiohttp
from openai import AsyncOpenAI
import random
from datasets import load_dataset
from tqdm.asyncio import tqdm
from typing import List, Tuple, Dict
import json
from asyncio import Semaphore
from collections import Counter
from functools import wraps
from collections import OrderedDict

# Configuration
POLICY_URL = 'https://rawsh--vllm-qwen-simpo-serve.modal.run/v1/'
PRM_URL = 'https://rawsh--mirrorqwen-prm-embedder-score-output.modal.run'
API_KEY = '9FF74944EED19865193F979942FB1'
BATCH_SIZE = 100  # Reduced batch size since we're doing multiple requests per question
MAX_RETRIES = 5
TIMEOUT = 20
MAX_CONCURRENT = 100
SAMPLES_PER_QUESTION = 10  # Default to single sample mode, override with CLI arg

# Cache decorator for PRM scores
def async_lru_cache(maxsize=2000):
    cache = OrderedDict()
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key not in cache:
                if len(cache) >= maxsize:
                    cache.popitem(last=False)
                cache[key] = await func(*args, **kwargs)
            return cache[key]
        return wrapper
    return decorator

class BatchProgress:
    def __init__(self, total_questions: int, samples_per_question: int):
        self.total = total_questions
        self.samples = samples_per_question
        self.correct_any = 0
        self.correct_best = 0
        self.correct_sc = 0
        self.processed = 0
        self.pbar = tqdm(total=total_questions, desc=self.get_description())
    
    def get_description(self) -> str:
        if self.processed == 0:
            return "Starting..."
        
        any_acc = (self.correct_any / self.processed) * 100
        if self.samples > 1:
            best_acc = (self.correct_best / self.processed) * 100
            sc_acc = (self.correct_sc / self.processed) * 100
            return f"Processed: {self.processed}/{self.total} | Any: {any_acc:.1f}% | Best: {best_acc:.1f}% | SC: {sc_acc:.1f}%"
        else:
            return f"Processed: {self.processed}/{self.total} | Accuracy: {any_acc:.1f}%"
    
    def update(self, any_correct: bool, best_correct: bool = None, sc_correct: bool = None):
        self.processed += 1
        if any_correct:
            self.correct_any += 1
        if best_correct:
            self.correct_best += 1
        if sc_correct:
            self.correct_sc += 1
        self.pbar.update(1)
        self.pbar.set_description(self.get_description())
    
    def close(self):
        self.pbar.close()
        if self.processed > 0:
            any_acc = (self.correct_any / self.processed) * 100
            print(f"\nFinal Results:")
            print(f"Total Questions: {self.processed}")
            print(f"Single Sample Accuracy: {any_acc:.2f}%")
            
            if self.samples > 1:
                best_acc = (self.correct_best / self.processed) * 100
                sc_acc = (self.correct_sc / self.processed) * 100
                print(f"Best-of-{self.samples} Accuracy: {best_acc:.2f}%")
                print(f"Self-Consistency Accuracy: {sc_acc:.2f}%")

async def retry_with_exponential_backoff(func, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=TIMEOUT)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            delay = min(1.5 ** attempt + random.random(), 10)
            await asyncio.sleep(delay)

@async_lru_cache(maxsize=1000)
async def get_prm_score(completion: str, session: aiohttp.ClientSession) -> float:
    """Get the PRM score for a completion."""
    async with session.post(PRM_URL, json={"prompt": completion}) as response:
        result = await response.json()
    return float(result['score'])

async def generate_completion(
    question: str,
    client: AsyncOpenAI,
    semaphore: Semaphore
) -> str:
    """Generate a single completion."""
    async with semaphore:
        response = await client.completions.create(
            model="mirrorqwen2.5-0.5b-SimPO-3",
            prompt=question,
            max_tokens=1500,
            temperature=0.8
        )
        return response.choices[0].text.strip()

async def evaluate_question(
    question: str,
    answer: str,
    client: AsyncOpenAI,
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    samples_per_question: int
) -> Dict:
    """Evaluate a question with single or multiple samples."""
    try:
        # Generate completions
        completions = []
        for _ in range(samples_per_question):
            completion = await retry_with_exponential_backoff(
                generate_completion, question, client, semaphore
            )
            completions.append(completion)
        
        # For single sample mode, return simpler result
        if samples_per_question == 1:
            is_correct = fr"\boxed{{{answer}}}" in completions[0]
            return {
                "question": question,
                "expected_answer": answer,
                "completion": completions[0],
                "correct": is_correct
            }
        
        # For multi-sample mode, evaluate with PRM
        scores = []
        for completion in completions:
            score = await retry_with_exponential_backoff(
                get_prm_score, completion, session
            )
            scores.append(score)
        
        # Evaluate correctness and extract answers
        is_correct = []
        extracted_answers = []
        for completion in completions:
            correct = fr"\boxed{{{answer}}}" in completion
            is_correct.append(correct)
            
            # Extract answer for self-consistency
            if r"\boxed{" in completion:
                extracted = completion.split(r"\boxed{")[1].split("}")[0]
                extracted_answers.append(extracted)
        
        # Find best completion by PRM score
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        
        # Calculate self-consistency
        answer_counts = Counter(extracted_answers)
        most_common_answer = answer_counts.most_common(1)[0][0] if answer_counts else None
        is_sc_correct = most_common_answer == answer if most_common_answer else False
        
        return {
            "question": question,
            "expected_answer": answer,
            "completions": [
                {
                    "text": compl,
                    "score": score,
                    "correct": corr
                }
                for compl, score, corr in zip(completions, scores, is_correct)
            ],
            "best_completion": {
                "text": completions[best_idx],
                "score": scores[best_idx],
                "correct": is_correct[best_idx]
            },
            "statistics": {
                "any_correct": any(is_correct),
                "best_correct": is_correct[best_idx],
                "self_consistency_correct": is_sc_correct,
                "unique_answers": len(answer_counts),
                "most_common_answer": most_common_answer,
                "most_common_count": answer_counts.most_common(1)[0][1] if answer_counts else 0
            }
        }
        
    except Exception as e:
        return {
            "question": question,
            "expected_answer": answer,
            "error": str(e)
        }

async def process_batch(
    batch: List[Tuple[str, str]],
    client: AsyncOpenAI,
    session: aiohttp.ClientSession,
    progress: BatchProgress,
    semaphore: Semaphore,
    samples_per_question: int
) -> List[dict]:
    """Process a batch of questions concurrently."""
    tasks = []
    for question, answer in batch:
        tasks.append(
            evaluate_question(
                question, answer, client, session, semaphore, samples_per_question
            )
        )
    
    results = await asyncio.gather(*tasks)
    
    # Update progress based on mode
    for result in results:
        if "error" not in result:
            if samples_per_question == 1:
                progress.update(result["correct"])
            else:
                progress.update(
                    result["statistics"]["any_correct"],
                    result["statistics"]["best_correct"],
                    result["statistics"]["self_consistency_correct"]
                )
    
    return results

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10,
                    help="Number of samples per question (default: 1)")
    parser.add_argument("--num-questions", type=int, default=200,
                    help="Number of questions to evaluate (default: 200)")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load and preprocess dataset
    gsm8k = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=42)
    questions = [(ex["question"], ex["answer"].split("\n#### ")[-1].strip()) 
                for ex in gsm8k]
    # questions = random.sample(questions, args.num_questions)
    
    # Initialize API client and semaphore
    client = AsyncOpenAI(base_url=POLICY_URL, api_key=API_KEY)
    semaphore = Semaphore(MAX_CONCURRENT)
    
    # Initialize progress tracker
    progress = BatchProgress(len(questions), args.samples)
    
    # Process in batches
    all_results = []
    
    # Create session only if needed (multi-sample mode)
    if args.samples > 1:
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(questions), BATCH_SIZE):
                batch = questions[i:i + BATCH_SIZE]
                results = await process_batch(
                    batch, client, session, progress, semaphore, args.samples
                )
                all_results.extend(results)
    else:
        # Use None for session in single-sample mode
        for i in range(0, len(questions), BATCH_SIZE):
            batch = questions[i:i + BATCH_SIZE]
            results = await process_batch(
                batch, client, None, progress, semaphore, args.samples
            )
            all_results.extend(results)
    
    # Save results
    suffix = f"{args.samples}samples" if args.samples > 1 else "single"
    filename = f"sampling_results_{suffix}.jsonl"
    with open(filename, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    
    progress.close()

if __name__ == "__main__":
    asyncio.run(main())