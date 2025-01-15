import asyncio
from openai import AsyncOpenAI
from datasets import load_dataset
from tqdm import tqdm
import random
import aiohttp
from typing import Optional, Tuple

# Configuration
POLICY_URL = 'https://rawsh--vllm-qwq-distill-serve.modal.run/v1/'
API_KEY = '9FF74944EED19865193F979942FB1'
MODEL_NAME = 'q1-Qwen2.5-0.5b'
# MODEL_NAME = 'q1-Qwen2.5-0.5b-Instruct'
# MODEL_NAME = 'q1-Qwen2.5-Math-1.5B'
MAX_CONCURRENT_REQUESTS = 128
MAX_RETRIES = 2
# REQUEST_TIMEOUT = 300  # seconds
REQUEST_TIMEOUT = 3000  # seconds
MAX_TOKENS = 10000  # Maximum tokens per response

class TokenCounter:
    def __init__(self):
        self.total_tokens = 0
        self.total_possible = 0
        self.pbar = None

    def init_progress(self, total_problems):
        max_possible = total_problems * MAX_TOKENS
        self.pbar = tqdm(total=max_possible, desc="Token usage", unit="tokens", position=1, leave=True)
        self.total_possible = max_possible

    def update(self, tokens: int):
        self.total_tokens += tokens
        self.pbar.update(tokens)
        return tokens

    def get_stats(self):
        return f"{self.total_tokens}/{self.total_possible} tokens ({(self.total_tokens/self.total_possible)*100:.1f}%)"

async def solve_problem(client, question: str, token_counter: TokenCounter, max_tokens: int = MAX_TOKENS) -> Tuple[Optional[str], int]:
    """Generate a solution for a given math problem with retries, timeout, and token tracking."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with asyncio.timeout(REQUEST_TIMEOUT):
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": ""}
                    ],
                    max_tokens=max_tokens,
                    stop=["<|endoftext|>", "<|im_end|>"],
                    # temperature=0.0,
                    # temperature=0.1,
                    temperature=1.0,
                    extra_body={
                        # "repetition_penalty": 1.05,
                        # "repetition_penalty": 1.10,
                        # "top_p": 0.8,
                        # "top_k": 20,
                        "top_p": 1.0,
                        "top_k": -1,
                        # "frequency_penalty": 0.05,
                        # "presence_penalty": 0.05,
                        # "frequency_penalty": 0.15,
                        # "presence_penalty": 0.15,
                        # "min_p": 0.05,
                        "min_p": 0.00,
                    },
                    stream=True
                )
                
                full_response = ""
                total_tokens = 0
                
                # Print question at the start
                print(f"\033[K\nQ: {question}\nA: ", end="", flush=True)
                
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        # Print content as it arrives
                        # print(content.replace("\n", "\\n"), end="", flush=True)
                        token_counter.update(1)
                        total_tokens += 1
                
                # Print newline after response
                print("\n")
                return full_response.strip(), total_tokens
                
        except asyncio.TimeoutError:
            if attempt < MAX_RETRIES:
                print(f"\nTimeout on attempt {attempt + 1}, retrying...", flush=True)
                await asyncio.sleep(1 * (attempt + 1))
            else:
                print(f"\nTimeout after {MAX_RETRIES} retries for question: {question[:100]}...", flush=True)
                return "", 0
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"\nError on attempt {attempt + 1}: {e}, retrying...", flush=True)
                await asyncio.sleep(1 * (attempt + 1))
            else:
                print(f"\nError after {MAX_RETRIES} retries: {e}", flush=True)
                return "", 0
    return "", 0

async def process_problem(client, question, answer, semaphore, pbar, results, correct, token_counter):
    """Process a single problem with semaphore control and token tracking."""
    async with semaphore:
        solution, tokens_used = await solve_problem(client, question, token_counter)
        is_solved = is_correct(solution, answer)
        
        if is_solved:
            correct.value += 1
        
        result = {
            "question": question,
            "correct_answer": answer,
            "solution": solution,
            "is_correct": is_solved,
            "tokens_used": tokens_used
        }
        
        accuracy = (correct.value / (len(results) + 1)) * 100
        
        # Move cursor to bottom, update progress bar, then restore cursor
        print(f"\033[KTokens used: {tokens_used}")
        print("\033[K" + "-" * 40)
        
        # Update progress bar
        pbar.set_description(f"Solving problems [{accuracy:.1f}% correct] [Tokens: {token_counter.get_stats()}]")
        pbar.update(1)
        
        results.append(result)

def is_correct(solution, correct_answer):
    """Check if the solution contains the correct answer."""
    answer_segment = str(correct_answer).strip()
    sol_segment = solution.strip()[-300:]
    print("\nGROUND TRUTH", answer_segment, "\nLLM RESPONSE", sol_segment)
    print(f"{{{answer_segment}}}")
    return f"{{{answer_segment}}}" in sol_segment

class Counter:
    def __init__(self):
        self.value = 0

async def main():
    random.seed(42)
    
    client = AsyncOpenAI(base_url=POLICY_URL, api_key=API_KEY)
    
    # Clear screen and move cursor to top
    print("\033[2J\033[H", end="")
    
    await warmup_api(client)

    token_counter = TokenCounter()
    
    def process_aime(example):
        example["answer"] = str(int(example["answer"].strip()))
        return example
    
    print("Loading dataset...")
    dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
    dataset = dataset.map(process_aime)
    problems = [(example["problem"], example["answer"]) for example in dataset]
    problems = problems[5:10]
    # problems = problems[70:80]
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    results = []
    correct = Counter()
    
    # Initialize progress bars at the bottom
    token_counter.init_progress(len(problems))
    pbar = tqdm(total=len(problems), desc="Solving problems [0.0% correct]", position=0, leave=True)
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_problem(client, question, answer, semaphore, pbar, results, correct, token_counter)
            for question, answer in problems
        ]
        await asyncio.gather(*tasks)
    
    pbar.close()
    token_counter.pbar.close()
    
    # Print final results
    final_accuracy = (correct.value / len(problems)) * 100
    print(f"\nFinal Results:")
    print(f"Total Problems: {len(problems)}")
    print(f"Correct Solutions: {correct.value}")
    print(f"Final Accuracy: {final_accuracy:.2f}%")
    print(f"Token Usage: {token_counter.get_stats()}")
    
    with open("greedy_results.txt", "w") as f:
        f.write(f"Final Accuracy: {final_accuracy:.2f}%\n")
        f.write(f"Token Usage: {token_counter.get_stats()}\n\n")
        for result in results:
            f.write(f"Question: {result['question']}\n")
            f.write(f"Correct Answer: {result['correct_answer']}\n")
            f.write(f"Solution: {result['solution']}\n")
            f.write(f"Correct: {result['is_correct']}\n")
            f.write(f"Tokens Used: {result['tokens_used']}\n")
            f.write("-" * 80 + "\n")

async def warmup_api(client):
    """Warm up the API with a simple query and retry logic."""
    print("Warming up API...")
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with asyncio.timeout(REQUEST_TIMEOUT):
                completion = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                        {"role": "user", "content": "What is 5+45+4=?"}
                    ],
                    stop=["<|im_end|>"],
                    max_tokens=10,
                    stream=True
                )
                
                response = ""
                total_tokens = 0
                async for chunk in completion:
                    if chunk.choices[0].delta.content:
                        response += chunk.choices[0].delta.content
                        total_tokens += 1
                
                print("API warmup successful")
                return
        except (asyncio.TimeoutError, Exception) as e:
            if attempt < MAX_RETRIES:
                print(f"Warmup attempt {attempt + 1} failed: {e}, retrying...")
                await asyncio.sleep(1 * (attempt + 1))
            else:
                print(f"Warning: API warmup failed after {MAX_RETRIES} retries: {e}")

if __name__ == "__main__":
    asyncio.run(main())