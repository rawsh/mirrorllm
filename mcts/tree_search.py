import asyncio
import math
import aiohttp
import json
from openai import AsyncOpenAI
import time
import random  # Added for jitter in retry logic
from functools import wraps
from collections import OrderedDict
from asyncio import Semaphore, TimeoutError
from datasets import load_dataset
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

# URLs and configuration
POLICY_URL = 'https://rawsh--vllm-qwen-ft-serve.modal.run/v1/'
PRM_URL = 'https://rawsh--mirrorqwen-prm-embedder-score-output.modal.run'
API_KEY = '9FF74944EED19865193F979942FB1'

CONCURRENT_MCTS_SEMAPHORE = Semaphore(50)
POLICY_SEMAPHORE = Semaphore(1000)
PRM_SEMAPHORE = Semaphore(1000)

MAX_RETRIES = 25  # Increased from 10
TIMEOUT = 20    # Decreased from 30 to fail faster and retry

# Cache decorator and retry function
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

async def retry_with_timeout(func, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=TIMEOUT)
        except TimeoutError:
            if attempt == MAX_RETRIES - 1:
                raise
            # Exponential backoff with jitter
            delay = min(1.5 ** attempt + random.random(), 10)
            await asyncio.sleep(delay)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            # Exponential backoff with jitter for other errors
            delay = min(1.5 ** attempt + random.random(), 10)
            print(f"Attempt {attempt + 1} failed with error: {str(e)}. Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_value = 0
        self.prm_value = None

# Progress tracking class with added metrics
class MCTSProgress:
    def __init__(self, total_questions, iterations_per_question):
        self.total_questions = total_questions
        self.total_iterations = total_questions * iterations_per_question
        self.completed_questions = 0
        self.completed_iterations = 0
        self.correct_sc = 0  # Self-consistency correct count
        self.correct_any = 0  # Any-correct count
        self.correct_best = 0  # Best PRM path correct count
        self.total_actions = 0  # Global action counter
        self.total_terminal_questions = 0  # Questions with at least one terminal node
        
        # Single progress bar with dynamic description
        self.pbar = tqdm(total=self.total_iterations, 
                        desc=self.get_progress_description())
    
    def get_progress_description(self):
        completed_pct = (self.completed_iterations / self.total_iterations) * 100
        sc_pct = (self.correct_sc / max(1, self.total_terminal_questions)) * 100
        any_pct = (self.correct_any / max(1, self.total_terminal_questions)) * 100
        best_pct = (self.correct_best / max(1, self.total_terminal_questions)) * 100
        return (f"#Q: {self.completed_questions}/{self.total_questions} | "
                f"SC: {sc_pct:.1f}% | "
                f"ANY: {any_pct:.1f}% | "
                f"BEST: {best_pct:.1f}% | "
                f"Actions: {self.total_actions}")
    
    def increment_iteration(self):
        self.completed_iterations += 1
        self.pbar.update(1)
        self.pbar.set_description(self.get_progress_description())
    
    def complete_question(self, is_sc_correct, is_any_correct, is_best_correct, has_terminal_nodes):
        self.completed_questions += 1
        if has_terminal_nodes:
            self.total_terminal_questions += 1
            if is_sc_correct:
                self.correct_sc += 1
            if is_any_correct:
                self.correct_any += 1
            if is_best_correct:
                self.correct_best += 1
        self.pbar.set_description(self.get_progress_description())
    
    def close(self):
        # Print final statistics
        if self.total_terminal_questions > 0:
            sc_pct = (self.correct_sc / self.total_terminal_questions) * 100
            any_pct = (self.correct_any / self.total_terminal_questions) * 100
            best_pct = (self.correct_best / self.total_terminal_questions) * 100
            print(f"\nFinal Results:")
            print(f"Total Questions Processed: {self.completed_questions}")
            print(f"Questions with Terminal Nodes: {self.total_terminal_questions}")
            print(f"Self-Consistency Accuracy: {sc_pct:.2f}% ({self.correct_sc}/{self.total_terminal_questions})")
            print(f"Any-Correct Accuracy: {any_pct:.2f}% ({self.correct_any}/{self.total_terminal_questions})")
            print(f"Best-Path Accuracy: {best_pct:.2f}% ({self.correct_best}/{self.total_terminal_questions})")
            print(f"Total Actions Taken: {self.total_actions}")
        self.pbar.close()

def select(node):
    while node.children:
        if len(node.children) < len(get_possible_actions(node.state)):
            return node
        node = best_uct_child(node)
    return node

def best_uct_child(node):
    C = 1.41
    return max(
        node.children.values(),
        key=lambda child: (child.total_value / child.visits) + C * math.sqrt(math.log(node.visits) / child.visits)
    )

async def expand(node, client, session, progress_tracker):
    action = await retry_with_timeout(get_next_action, node.state, client)
    new_state = apply_action(node.state, action)
    child = Node(new_state, parent=node)
    node.children[action] = child
    progress_tracker.total_actions += 1
    return child

async def simulate(node, correct_answer, client, session, terminal_nodes, progress_tracker):
    state = node.state
    depth = 0
    max_depth = 10
    while depth < max_depth:
        is_term, is_corr = await retry_with_timeout(is_terminal, state, correct_answer, client, session)
        if is_term:
            terminal_nodes.add(state)
            break
        action = await retry_with_timeout(get_next_action, state, client)
        state = apply_action(state, action)
        progress_tracker.total_actions += 1
        depth += 1
    return await retry_with_timeout(evaluate_state, state, session)

def backpropagate(node, value):
    while node:
        node.visits += 1
        node.total_value += value
        node = node.parent

async def get_next_action(state, client):
    prompt = format_state_for_policy(state)
    async with POLICY_SEMAPHORE:
        response = await client.completions.create(
            model="rawsh/mirrorqwen2.5-0.5b-SFT",
            prompt=prompt,
            max_tokens=250,
            stop=["\n\n"],
            temperature=0.8
        )
    return response.choices[0].text.strip()

def is_correct(state, correct_answer):
    last_step = state.split("\n\n")[-1]
    return fr"\boxed{{{correct_answer}}}" in last_step

async def is_terminal(state, correct_answer, client, session):
    if is_correct(state, correct_answer):
        return True, True
    
    if state.count("\n\n") < 2:
        return False, False
    
    async with POLICY_SEMAPHORE:
        response = await client.completions.create(
            model="rawsh/mirrorqwen2.5-0.5b-SFT",
            prompt=state,
            max_tokens=1,
            stop=["\n\n"],
            temperature=0.3,
            logprobs=20,
        )
    
    first_token_top_logprobs = response.choices[0].logprobs.top_logprobs[0]
    if "" in first_token_top_logprobs:
        scaled = math.exp(first_token_top_logprobs[""])
        yes_bigger_than_no = True
        if "\n\n" in first_token_top_logprobs:
            scaled_no = math.exp(first_token_top_logprobs["\n\n"])
            yes_bigger_than_no = (scaled > scaled_no)

        threshold = 0.95
        terminal = (scaled >= threshold) and yes_bigger_than_no
        return terminal, False
    else:
        return False, False

@async_lru_cache(maxsize=1000)
async def evaluate_state(state, session):
    prompt = format_state_for_prm(state)
    async with PRM_SEMAPHORE:
        async with session.post(PRM_URL, json={"prompt": prompt}) as response:
            result = await response.json()
    return float(result['score'])

def apply_action(state, action):
    return f"{state}\n\n{action}"

def get_possible_actions(state):
    return range(3)

def format_state_for_policy(state):
    return f"{state}\n\n"

def format_state_for_prm(state):
    return state

def collect_leaf_nodes(node, leaf_nodes):
    if not node.children:
        leaf_nodes.append(node)
    else:
        for child in node.children.values():
            collect_leaf_nodes(child, leaf_nodes)

async def find_best_leaf_by_prm(node, session):
    leaf_nodes = []
    collect_leaf_nodes(node, leaf_nodes)
    tasks = []
    for leaf in leaf_nodes:
        if leaf.prm_value is None:
            tasks.append(evaluate_and_store_prm(leaf, session))
    await asyncio.gather(*tasks)
    return max(leaf_nodes, key=lambda leaf: leaf.prm_value if leaf.prm_value is not None else float('-inf'))

async def evaluate_and_store_prm(node, session):
    node.prm_value = await retry_with_timeout(evaluate_state, node.state, session)

async def mcts(root_state, correct_answer, num_iterations, session, progress_tracker):
    root = Node(root_state)
    client = AsyncOpenAI(base_url=POLICY_URL, api_key=API_KEY)
    terminal_nodes = set()
    
    for i in range(num_iterations):
        leaf = select(root)
        is_term, is_corr = await retry_with_timeout(is_terminal, leaf.state, correct_answer, client, session)
        
        if is_term:
            terminal_nodes.add(leaf.state)
        else:
            child = await retry_with_timeout(expand, leaf, client, session, progress_tracker)
            value = await retry_with_timeout(simulate, child, correct_answer, client, session, terminal_nodes, progress_tracker)
            backpropagate(child, value)
        
        progress_tracker.increment_iteration()
    
    return root, terminal_nodes

async def run_mcts(initial_state, correct_answer, num_iterations, session, progress_tracker):
    async with CONCURRENT_MCTS_SEMAPHORE:
        start_time = time.time()
        root, terminal_nodes = await mcts(initial_state, correct_answer, num_iterations, session, progress_tracker)
        end_time = time.time()
        
        best_leaf = await find_best_leaf_by_prm(root, session)
        
        terminal_paths = []
        terminal_correct_count = 0
        total_terminal_nodes = len(terminal_nodes)
        max_prm_score = float('-inf')
        best_prm_path_correct = False
        
        for node in terminal_nodes:
            score = await retry_with_timeout(evaluate_state, node, session)
            is_node_correct = is_correct(node, correct_answer)
            if is_node_correct:
                terminal_correct_count += 1
            if score > max_prm_score:
                max_prm_score = score
                best_prm_path_correct = is_node_correct
            terminal_paths.append({
                "final_state": node,
                "score": score,
                "correct": is_node_correct
            })
        
        is_best_correct = is_correct(best_leaf.state, correct_answer)
        
        # Calculate metrics using only terminal nodes
        # Self-consistency based on majority voting of terminal nodes (>50% correct)
        has_terminal_nodes = total_terminal_nodes > 0
        is_sc_correct = (terminal_correct_count > total_terminal_nodes / 2) if has_terminal_nodes else False
        is_any_correct = (terminal_correct_count > 0)  # Any-correct using terminal nodes
        
        result = {
            "question": initial_state,
            "correct_answer": correct_answer,
            "statistics": {
                "num_iterations": num_iterations,
                "execution_time": end_time - start_time,
                "total_terminal_nodes": total_terminal_nodes,
                "correct_terminal_nodes": terminal_correct_count,
                "self_consistency_correct": is_sc_correct,
                "any_correct": is_any_correct,
                "has_terminal_nodes": has_terminal_nodes,
                "best_prm_path_correct": best_prm_path_correct
            },
            "best_path": {
                "final_state": best_leaf.state,
                "score": best_leaf.prm_value,
                "correct": is_best_correct
            },
            "terminal_paths": terminal_paths
        }
        
        progress_tracker.complete_question(is_sc_correct, is_any_correct, best_prm_path_correct, has_terminal_nodes)
        return result

async def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    def process(example):
        example["answer"] = example["answer"].split("\n#### ")[-1].strip()
        return example

    gsm8k = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42)
    gsm8k = gsm8k.map(process, num_proc=24)
    initial_states = [(example["question"], example["answer"]) for example in gsm8k]
    
    # Sample 100 questions
    sample = False
    if sample:
        initial_states = random.sample(initial_states, 10)
    
    num_iterations = 10

    # Initialize progress tracker
    progress_tracker = MCTSProgress(len(initial_states), num_iterations)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for state, answer in initial_states:
            tasks.append(run_mcts(state, answer, num_iterations, session, progress_tracker))
        
        results = await asyncio.gather(*tasks)
    
    progress_tracker.close()
    
    # Calculate and print final statistics
    total_questions = len(results)
    sc_correct = sum(1 for r in results if r["statistics"]["self_consistency_correct"])
    any_correct = sum(1 for r in results if r["statistics"]["any_correct"])
    
    print(f"\nFinal Statistics:")
    print(f"Total Questions: {total_questions}")
    print(f"Self-Consistency Accuracy: {(sc_correct/total_questions)*100:.2f}%")
    print(f"Any-Correct Accuracy: {(any_correct/total_questions)*100:.2f}%")
    
    # Write results
    with open("mcts_results.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    asyncio.run(main())