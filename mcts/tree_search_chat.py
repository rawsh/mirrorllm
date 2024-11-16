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
# POLICY_URL = 'https://rawsh--vllm-qwen-ft-serve.modal.run/v1/'
# POLICY_MODEL_NAME = 'mirrorqwen2.5-0.5b-SimPO-3'
# POLICY_MODEL_NAME = 'mirrorqwen2.5-0.5b-SimPO-0'
# POLICY_MODEL_NAME = 'mirrorqwen2.5-0.5b-SFT'
# POLICY_MODEL_NAME = 'mirrorqwen2.5-0.5b-ORPO-1'
# POLICY_MODEL_NAME = 'mirrorqwen2.5-0.5b-ORPO-2'
# POLICY_MODEL_NAME = 'mirrorqwen2.5-0.5b-ORPO-3'
POLICY_MODEL_NAME = 'MetaMath-Qwen2.5-0.5b'
# POLICY_URL = 'https://rawsh--vllm-qwen-simpo-serve.modal.run/v1/'
# POLICY_URL = 'https://rawsh--vllm-qwen-base-serve.modal.run/v1/'
# POLICY_URL = 'https://rawsh--vllm-qwen-orpo-serve.modal.run/v1/'
POLICY_URL = 'https://rawsh--vllm-qwen-metamath-serve.modal.run/v1/'
# PRM_URL = 'https://rawsh--mirrorqwen-prm-embedder-score-output.modal.run'
PRM_URL = 'https://rawsh--mirrorqwen-prm-st-embedder-score-output.modal.run'
API_KEY = '9FF74944EED19865193F979942FB1'

CONCURRENT_MCTS_SEMAPHORE = Semaphore(20)
POLICY_SEMAPHORE = Semaphore(1000)
PRM_SEMAPHORE = Semaphore(1000)

MAX_RETRIES = 20  # Increased from 10s
TIMEOUT = 10   # Decreased from 30 to fail faster and retry

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
            return await asyncio.wait_for(func(*args, **kwargs), timeout=TIMEOUT * max(1, attempt / 10 ))
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
        self.iterations_per_question = iterations_per_question
        self.completed_iterations = 0
        self.correct_sc = 0  # Self-consistency correct count
        self.correct_any = 0  # Any-correct count
        self.correct_best = 0  # Best PRM path correct count
        self.total_actions = 0  # Global action counter
        self.questions_with_terminal = 0  # Questions with at least one terminal path
        self.fully_completed_questions = 0  # Questions that completed all iterations
        
        # Single progress bar with dynamic description
        self.pbar = tqdm(total=self.total_iterations, 
                        desc=self.get_progress_description())
    
    def get_progress_description(self):
        sc_pct = (self.correct_sc / max(1, self.fully_completed_questions)) * 100
        any_pct = (self.correct_any / max(1, self.fully_completed_questions)) * 100
        best_pct = (self.correct_best / max(1, self.fully_completed_questions)) * 100
        q_pct = (self.questions_with_terminal / self.total_questions) * 100
        return (f"#Q ({self.questions_with_terminal}/{self.total_questions}): {q_pct:.0f}% | "
                f"SC: {sc_pct:.1f}% | "
                f"ANY: {any_pct:.1f}% | "
                f"BEST: {best_pct:.1f}% | "
                f"Actions: {self.total_actions}")
    
    def increment_iteration(self):
        self.completed_iterations += 1
        self.pbar.update(1)
        # No need to update description here
    
    def complete_question(self, is_sc_correct, is_any_correct, is_best_correct, is_fully_completed, has_terminal_nodes):
        if has_terminal_nodes:
            self.questions_with_terminal += 1
        if is_fully_completed:
            self.fully_completed_questions += 1
            if is_sc_correct:
                self.correct_sc += 1
            if is_any_correct:
                self.correct_any += 1
            if is_best_correct:
                self.correct_best += 1
        self.pbar.set_description(self.get_progress_description())
    
    def close(self):
        # Print final statistics
        if self.fully_completed_questions > 0:
            sc_pct = (self.correct_sc / self.fully_completed_questions) * 100
            any_pct = (self.correct_any / self.fully_completed_questions) * 100
            best_pct = (self.correct_best / self.fully_completed_questions) * 100
            print(f"\nFinal Results:")
            print(f"Questions with Terminal Paths: {self.questions_with_terminal}")
            print(f"Fully Completed Questions: {self.fully_completed_questions}")
            print(f"Self-Consistency Accuracy: {sc_pct:.2f}% ({self.correct_sc}/{self.fully_completed_questions})")
            print(f"Any-Correct Accuracy: {any_pct:.2f}% ({self.correct_any}/{self.fully_completed_questions})")
            print(f"Best-Path Accuracy: {best_pct:.2f}% ({self.correct_best}/{self.fully_completed_questions})")
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
        steps = prompt.split("\n\n")
        question = steps[0]
        answer = None
        if len(steps) > 0:
            answer = "\n\n".join(steps[1:])

        messages = [
            {"role": "user", "content": question}
        ]
        if answer is not None:
            messages.append({"role": "assistant", "content": answer})
        
        response = await client.chat.completions.create(
            model=POLICY_MODEL_NAME,
            messages=messages,
            max_tokens=250,
            stop=["\n\n"],
            temperature=0.8,
        )
    # return response.choices[0].text.strip()
    return response.choices[0].message.content.strip()

def is_correct(state, correct_answer):
    last_step = state.split("\n\n")[-1]
    return fr"\boxed{{{correct_answer}}}" in last_step

async def is_terminal(state, correct_answer, client, session):
    if is_correct(state, correct_answer):
        return True, True
    
    if state.count("\n\n") < 2:
        return False, False
    
    async with POLICY_SEMAPHORE:
        steps = state.split("\n\n")
        question = steps[0]
        answer = None
        if len(steps) > 0:
            answer = "\n\n".join(steps[1:])

        messages = [
            {"role": "user", "content": question}
        ]
        if answer is not None:
            messages.append({"role": "assistant", "content": answer})
        
        response = await client.chat.completions.create(
            model=POLICY_MODEL_NAME,
            messages=messages,
            max_tokens=1,
            stop=["\n\n"],
            temperature=0.8,
            logprobs=True,
            top_logprobs=20
        )
        # response = await client.completions.create(
        #     model=POLICY_MODEL_NAME,
        #     prompt=state,
        #     max_tokens=1,
        #     stop=["\n\n"],
        #     temperature=0.3,
        #     logprobs=20,
        # )

    first_token_top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    first_token_top_logprobs_map = dict()
    for token_logprob in first_token_top_logprobs:
        first_token_top_logprobs_map[token_logprob.token] = token_logprob.logprob

    if "" in first_token_top_logprobs_map:
        scaled = math.exp(first_token_top_logprobs_map[""])
        yes_bigger_than_no = True
        if "\n\n" in first_token_top_logprobs_map:
            scaled_no = math.exp(first_token_top_logprobs_map["\n\n"])
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
        answers = {}  # Track answer frequencies
        max_prm_score = float('-inf')
        best_prm_path_correct = False
        terminal_correct_count = 0  # Add this counter
        
        for node in terminal_nodes:
            score = await retry_with_timeout(evaluate_state, node, session)
            is_node_correct = is_correct(node, correct_answer)
            if is_node_correct:
                terminal_correct_count += 1  # Increment counter
            
            # Extract answer from the node
            last_step = node.split("\n\n")[-1]
            if r"\boxed{" in last_step:
                answer = last_step.split(r"\boxed{")[1].split("}")[0]
                answers[answer] = answers.get(answer, 0) + 1
            
            if score > max_prm_score:
                max_prm_score = score
                best_prm_path_correct = is_node_correct
                
            terminal_paths.append({
                "final_state": node,
                "score": score,
                "correct": is_node_correct
            })
        
        is_best_correct = is_correct(best_leaf.state, correct_answer)
        
        # Calculate SC using most common answer
        has_terminal_nodes = len(terminal_nodes) > 0
        is_sc_correct = False
        if has_terminal_nodes and answers:
            most_common_answer = max(answers.items(), key=lambda x: x[1])[0]
            is_sc_correct = any(p["correct"] and most_common_answer == p["final_state"].split(r"\boxed{")[1].split("}")[0] 
                              for p in terminal_paths)
        
        is_any_correct = any(p["correct"] for p in terminal_paths)
        is_fully_completed = len(terminal_nodes) > 0 and num_iterations == progress_tracker.iterations_per_question
        
        result = {
            "question": initial_state,
            "correct_answer": correct_answer,
            "statistics": {
                "num_iterations": num_iterations,
                "execution_time": end_time - start_time,
                "total_terminal_nodes": len(terminal_nodes),  # Use len() directly
                "correct_terminal_nodes": terminal_correct_count,
                "self_consistency_correct": is_sc_correct,
                "any_correct": is_any_correct,
                "has_terminal_nodes": has_terminal_nodes,
                "best_prm_path_correct": best_prm_path_correct,
                "fully_completed": is_fully_completed
            },
            "best_path": {
                "final_state": best_leaf.state,
                "score": best_leaf.prm_value,
                "correct": is_best_correct
            },
            "terminal_paths": terminal_paths
        }
        
        progress_tracker.complete_question(is_sc_correct, is_any_correct, best_prm_path_correct, is_fully_completed, has_terminal_nodes)
        return result

async def main():
    # Set random seed for reproducibility
    random.seed(0) # eval set - all models
    # random.seed(42) # st 0
    # random.seed(4242) # st 1
    # random.seed(424242) # st 2
    # random.seed(42424242) # st 3
    
    def process(example):
        example["answer"] = example["answer"].split("\n#### ")[-1].strip()
        return example

    gsm8k = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=42)
    # gsm8k = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42)
    gsm8k = gsm8k.map(process, num_proc=24)
    initial_states = [(example["question"], example["answer"]) for example in gsm8k]

    # SAMPLE 200 QUESTIONS - SELF TRAINING
    initial_states = random.sample(initial_states, 100)
    # initial_states = random.sample(initial_states, 1000)
    num_iterations = 10

    print("cold starting policy vllm + prm api")

    # warm up the chat API
    client = AsyncOpenAI(base_url=POLICY_URL, api_key=API_KEY)
    completion_promise = client.chat.completions.create(
        model=POLICY_MODEL_NAME,
        messages=[
            {"role": "user", "content": "Which is larger 9.11 or 9.9? Respond with just the answer."}
        ],
        # max_tokens=3,
        # stop=["\n\n"],
        stop=["<|endoftext|>"],
        temperature=0.8,
        # Note: logprobs is not available in chat format
    )
    # res = await completion_promise
    # print(res)
    # return

    async with aiohttp.ClientSession() as session:
        # warm up PRM api
        async with session.post(PRM_URL, json={"prompt": "TEST"}) as response:
            prm_promise = response.json()
            prm_score = await prm_promise
            assert('score' in prm_score)
            print("warmed up PRM api")

        completion = await completion_promise
        assert(len(completion.choices) == 1)
        print(completion.choices[0])
        print("warmed up vllm")
    

        # Initialize progress tracker
        progress_tracker = MCTSProgress(len(initial_states), num_iterations)

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