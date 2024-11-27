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

POLICY_MODEL_NAME = 'MetaMath-Qwen2.5-0.5b'
POLICY_URL = 'https://rawsh--vllm-qwen-metamath-serve.modal.run/v1/'
PRM_URL = 'https://rawsh--vllm-qwen-prm-serve.modal.run/v1/'
PRM_MODEL_NAME = 'MetaMath-Qwen2.5-0.5b-PRM'
API_KEY = '9FF74944EED19865193F979942FB1'

# Global clients
POLICY_CLIENT = AsyncOpenAI(base_url=POLICY_URL, api_key=API_KEY)
PRM_CLIENT = AsyncOpenAI(base_url=PRM_URL, api_key=API_KEY)

# More aggressive semaphore limits
CONCURRENT_MCTS_SEMAPHORE = Semaphore(200)
POLICY_SEMAPHORE = Semaphore(1000)
PRM_SEMAPHORE = Semaphore(1000)

# More aggressive retry settings
MAX_RETRIES = 10
TIMEOUT = 45

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
        except (TimeoutError, Exception) as e:
            print(f"WARNING: timeout during attempt {attempt}")
            if attempt == MAX_RETRIES - 1:
                raise
            # Faster backoff
            delay = min(0.1 * (attempt + 1), 1.0)
            await asyncio.sleep(delay)

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_value = 0
        self.prm_value = None
        self.step_scores = None
        self.virtual_loss = 0

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    async def evaluate(self):
        """Evaluate this node, caching step scores for reuse by children."""
        if self.step_scores is None:
            # Get parts of the solution
            parts = self.state.split("\n\n")
            if len(parts) < 2:
                print("WARNING: len(parts) < 2")
                self.step_scores = []
                self.prm_value = 1e-10
                return self.prm_value

            new_prefix = self.state
            async with PRM_SEMAPHORE:
                new_score = await evaluate_step(new_prefix)
                self.prm_value = new_score

        return self.prm_value

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
    
    def update_progress(self):
        self.pbar.set_description(self.get_progress_description())

    def increment_iteration(self):
        self.completed_iterations += 1
        self.pbar.update(1)
        # No need to update description here
        self.pbar.set_description(self.get_progress_description())

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

VIRTUAL_LOSS = 1.0

async def select(node):
    while node.children:
        if len(node.children) < len(get_possible_actions(node.state)):
            # Apply virtual loss before expansion
            node.virtual_loss += VIRTUAL_LOSS
            return node
        node = await best_uct_child(node)
    # Apply virtual loss at the leaf node
    node.virtual_loss += VIRTUAL_LOSS
    return node

async def best_uct_child(node):
    C = 1.41
    best_value = float('-inf')
    best_child = None
    parent_visits = max(1, node.visits + node.virtual_loss)
    ln_parent = math.log(parent_visits)
    for child in node.children.values():
        child_visits = child.visits + child.virtual_loss
        if child_visits < 1:
            child_visits = 1
        exploit = child.total_value / child_visits
        explore = C * math.sqrt(ln_parent / child_visits)
        uct_value = exploit + explore
        if uct_value > best_value:
            best_value = uct_value
            best_child = child
    return best_child

async def expand(node, client, session, progress_tracker):
    async with POLICY_SEMAPHORE:
        action = await retry_with_timeout(get_next_action, node.state, client)
    new_state = apply_action(node.state, action)
    child = Node(new_state, parent=node)
    node.children[action] = child
    progress_tracker.total_actions += 1
    progress_tracker.update_progress()
    return child

async def simulate(node, correct_answer, client, session, terminal_nodes, progress_tracker):
    current_node = node
    depth = 0
    max_depth = 10

    while depth < max_depth:
        if current_node in terminal_nodes:
            break
        
        async with POLICY_SEMAPHORE:
            action, is_term = await retry_with_timeout(get_next_action, current_node.state, client)
        
        new_state = apply_action(current_node.state, action)
        child_node = Node(new_state, parent=current_node)
        progress_tracker.total_actions += 1

        if is_term or is_correct(new_state, correct_answer):
            terminal_nodes.add(child_node)
            current_node = child_node
            break

        current_node = child_node
        depth += 1

    return await retry_with_timeout(evaluate_state, current_node.state, session)

async def backpropagate(node, value):
    while node:
        # Remove virtual loss
        node.virtual_loss -= VIRTUAL_LOSS

        # Update with actual value
        node.visits += 1
        node.total_value += value
        node = node.parent

async def get_next_action(state, client):
    steps = state.split("\n\n")
    question = steps[0]
    answer = "\n\n".join(steps[1:]) if len(steps) > 1 else None

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": question}
    ]
    if answer:
        messages.append({"role": "assistant", "content": answer + "\n\n"})
    else:
        messages.append({"role": "assistant", "content": ""})

    response = await client.chat.completions.create(
        timeout=TIMEOUT,
        model=POLICY_MODEL_NAME,
        messages=messages,
        max_tokens=150,
        stop=["<|endoftext|>", "<|im_end|>", "\n\n"],
        # temperature=0.7,
        temperature=1.0,
        extra_body={
            "repetition_penalty": 1.05,
            "top_p": 0.8,
            "top_k": 20,
            "frequency_penalty": 0.05,
            "presence_penalty": 0.05,
        }
    )
    content = response.choices[0].message.content.strip()

    # Determine if the assistant has stopped generating due to the stop sequence
    is_term = (response.choices[0].finish_reason == 'stop' and \
            response.choices[0].stop_reason != '\n\n')
    return content, is_term

def is_correct(state, correct_answer):
    last_step = state.split("\n\n")[-1]
    # Normalize the strings for comparison
    return correct_answer.strip() in last_step.strip()

# Create single global client
PRM_CLIENT = AsyncOpenAI(base_url=PRM_URL, api_key=API_KEY)

# Cache for step scores
step_scores_cache = {}

@async_lru_cache(maxsize=10000)
async def evaluate_step(step_prefix: str) -> float:
    """Evaluate a single solution step using PRM."""
    steps = step_prefix.split("\n\n")
    question = steps[0]
    curr_step = steps[-1]

    # Format messages for just this step evaluation
    if len(steps) == 2:
        messages = [
            {"role": "user", "content": f"{question} Step 1: {curr_step}"}
        ]
    else:
        messages = [
            {"role": "user", "content": f"{question} Step 1: {steps[1]}"}
        ]
        for i, step in enumerate(steps[2:-1], start=2):
            messages.extend([
                {"role": "assistant", "content": "+"},
                {"role": "user", "content": f"Step {i}: {step}"}
            ])
        curr_step_num = len(steps)-1
        messages.extend([
            {"role": "assistant", "content": "+"},
            {"role": "user", "content": f"Step {curr_step_num}: {curr_step}"}
        ])

    response = await PRM_CLIENT.chat.completions.create(
        timeout=TIMEOUT,
        model=PRM_MODEL_NAME,
        messages=messages,
        max_tokens=1,
        temperature=0.0,
        logprobs=True,
        top_logprobs=20,
        extra_body={
            "repetition_penalty": 1.05,
            "top_p": 0.8,
            "top_k": 20,
            "frequency_penalty": 0.05,
            "presence_penalty": 0.05,
            "add_generation_prompt": True,
        }
    )

    logprobs = response.choices[0].logprobs.content[0].top_logprobs
    # Get raw probabilities, defaulting to very small number if token not found
    prob_plus = next((math.exp(lp.logprob) for lp in logprobs if lp.token == "+"), 1e-10)

    # Normalize between + and -
    final_prob = prob_plus
    return final_prob

async def evaluate_state(state, session):
    """Simplified evaluate_state that creates a temporary node for evaluation."""
    node = Node(state)
    score =  await node.evaluate()
    return score

def apply_action(state, action):
    return f"{state}\n\n{action}"

def get_possible_actions(state):
    return range(3)

def collect_leaf_nodes(node, leaf_nodes):
    if not node.children:
        leaf_nodes.append(node)
    else:
        for child in node.children.values():
            collect_leaf_nodes(child, leaf_nodes)

async def find_best_leaf_by_prm(node, session):
    leaf_nodes = []
    collect_leaf_nodes(node, leaf_nodes)

    # Evaluate all leaves in parallel
    await asyncio.gather(*[leaf.evaluate() for leaf in leaf_nodes])
    
    return max(leaf_nodes, key=lambda leaf: leaf.prm_value)


async def mcts_iteration(root, correct_answer, client, session, terminal_nodes, progress_tracker):
    async with CONCURRENT_MCTS_SEMAPHORE:
        leaf = await select(root)

        if leaf.state in terminal_nodes:
            return

        action, is_term = await retry_with_timeout(get_next_action, leaf.state, client)
        new_state = apply_action(leaf.state, action)
        child = Node(new_state, parent=leaf)
        leaf.children[action] = child
        progress_tracker.total_actions += 1

        # Check if the last step contains the correct answer
        if is_term or is_correct(new_state, correct_answer):
            terminal_nodes.add(child)
            value = await retry_with_timeout(evaluate_state, child.state, session)
            await backpropagate(child, value)
        else:
            value = await simulate(
                child, correct_answer, client, session, terminal_nodes, progress_tracker
            )
            await backpropagate(child, value)

        progress_tracker.increment_iteration()

async def mcts(root_state, correct_answer, num_iterations, session, progress_tracker):
    root = Node(root_state)
    client = AsyncOpenAI(base_url=POLICY_URL, api_key=API_KEY)
    terminal_nodes = set()

    tasks = [
        mcts_iteration(root, correct_answer, client, session, terminal_nodes, progress_tracker)
        for _ in range(num_iterations)
    ]
    await asyncio.gather(*tasks)

    return root, terminal_nodes

async def run_mcts(initial_state, correct_answer, num_iterations, session, progress_tracker):
    # async with CONCURRENT_MCTS_SEMAPHORE:
    start_time = time.time()
    root, terminal_nodes = await mcts(initial_state, correct_answer, num_iterations, session, progress_tracker)
    end_time = time.time()

    best_leaf = await find_best_leaf_by_prm(root, session)

    terminal_paths = []
    answers = {}
    max_prm_score = float('-inf')
    best_prm_path_correct = False
    terminal_correct_count = 0

    for node in terminal_nodes:
        await node.evaluate()
        is_node_correct = is_correct(node.state, correct_answer)
        if is_node_correct:
            terminal_correct_count += 1

        last_step = node.state.split("\n\n")[-1]
        answer = last_step.strip()
        answers[answer] = answers.get(answer, 0) + 1

        if node.prm_value > max_prm_score:
            max_prm_score = node.prm_value
            best_prm_path_correct = is_node_correct

        terminal_paths.append({
            "final_state": node.state,
            "score": node.prm_value,
            "correct": is_node_correct
        })

    is_best_correct = is_correct(best_leaf.state, correct_answer)

    # Determine self-consistency correctness
    is_sc_correct = False
    if answers:
        most_common_answer = max(answers.items(), key=lambda x: x[1])[0]
        is_sc_correct = correct_answer.strip() in most_common_answer

    is_any_correct = terminal_correct_count > 0
    is_fully_completed = num_iterations == progress_tracker.iterations_per_question

    result = {
        "question": initial_state,
        "correct_answer": correct_answer,
        "statistics": {
            "num_iterations": num_iterations,
            "execution_time": end_time - start_time,
            "total_terminal_nodes": len(terminal_nodes),
            "correct_terminal_nodes": terminal_correct_count,
            "self_consistency_correct": is_sc_correct,
            "any_correct": is_any_correct,
            "has_terminal_nodes": len(terminal_nodes) > 0,
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

    progress_tracker.complete_question(
        is_sc_correct, is_any_correct, best_prm_path_correct, is_fully_completed, len(terminal_nodes) > 0
    )
    return result

# [info]
# greedy base sft:     57.0%
# greedy base:         41.6%
# greedy instruct:     49.6%
# greedy math-1.5b:    71.3%
# greedy math-1.5b-it: 84.8%

# [policy temp 0.7, prm greedy]
# mcts=1:   72% best (72% pass, 72% sc)
# mcts=2:   71% best (76% pass, 67% sc)
# mcts=3:   71% best (83% pass, 67% sc)
# mcts=4:   74% best (85% pass, 67% sc)
# mcts=5:   73% best (90% pass, 66% sc)
# mcts=6:   75% best (92% pass, 73% sc)
# mcts=7:   76% best (91% pass, 69% sc)
# mcts=8:   75% best (91% pass, 72% sc)
# mcts=9:   78% best (91% pass, 73% sc)
# mcts=10:  82% best (92% pass)
# mcts=20:  83% best (94% pass)
# mcts=30:  84% best (96% pass)
# mcts=40:  85% best (97% pass)
# mcts=50:  88% best (95% pass) 
# mcts=100: 86% best (98% pass)
# mcts=200: 84% best (99% pass)

# [policy temp 1.0, prm greedy]
# wtf. tmp 1.0??? that big a difference?
# mcts=4:   80% best (87% pass, 75% sc)
# mcts=10:  78% best (90% pass, 69% sc)
# mcts=20:  83% best (94% pass, 71% sc)

async def main():
    # Set random seed for reproducibility
    random.seed(0)

    def process(example):
        example["answer"] = example["answer"].split("\n#### ")[-1].strip()
        return example

    gsm8k = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=42)
    gsm8k = gsm8k.map(process, num_proc=24)
    initial_states = [(example["question"], example["answer"]) for example in gsm8k]
    initial_states = random.sample(initial_states, 100)
    num_iterations = 50

    print("cold starting policy vllm + prm api")

    # warm up the chat API
    client = AsyncOpenAI(base_url=POLICY_URL, api_key=API_KEY)
    
    async with aiohttp.ClientSession() as session:
        # First warm up vLLM API
        completion_promise = client.chat.completions.create(
            model=POLICY_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": "What is 5+45+4=?"}
            ],
            stop=["<|im_end|>"],
            temperature=0.3,
            max_tokens=200,
        )

        # Then warm up PRM api
        prm_client = AsyncOpenAI(base_url=PRM_URL, api_key=API_KEY)
        prm_response = await prm_client.chat.completions.create(
            model=PRM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": "1+1=2"},
                {"role": "assistant", "content": "+"},
                {"role": "user", "content": "Next, 2+2=4"}
            ],
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20
        )
        assert(len(prm_response.choices) == 1)
        print("warmed up PRM api")

        completion = await completion_promise
        assert(len(completion.choices) == 1)
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
    best_correct = sum(1 for r in results if r["statistics"]["best_prm_path_correct"])
    sc_correct = sum(1 for r in results if r["statistics"]["self_consistency_correct"])
    any_correct = sum(1 for r in results if r["statistics"]["any_correct"])

    print(f"\nFinal Statistics:")
    print(f"Total Questions: {total_questions}")
    print(f"Best-PRM Accuracy: {(best_correct/total_questions)*100:.2f}%")
    print(f"Self-Consistency Accuracy: {(sc_correct/total_questions)*100:.2f}%")
    print(f"Any-Correct Accuracy: {(any_correct/total_questions)*100:.2f}%")

    # Write results to both JSONL and TXT
    with open("mcts_results.jsonl", "w") as f_json, open("mcts_results.txt", "w") as f_txt:
        # Write summary stats to TXT
        f_txt.write("Final Statistics\n")
        f_txt.write(f"Total Questions: {total_questions}\n")
        f_txt.write(f"Best-PRM Accuracy: {(best_correct/total_questions)*100:.2f}%\n")
        f_txt.write(f"Self-Consistency Accuracy: {(sc_correct/total_questions)*100:.2f}%\n")
        f_txt.write(f"Any-Correct Accuracy: {(any_correct/total_questions)*100:.2f}%\n\n")
        
        # Write detailed results
        for result in results:
            f_json.write(json.dumps(result) + "\n")
            f_txt.write(f"Question {result['question']}:\n")
            f_txt.write(f"Best-PRM: {result['statistics']['best_prm_path_correct']}\n")
            f_txt.write(f"Self-Consistency: {result['statistics']['self_consistency_correct']}\n")
            f_txt.write(f"Any-Correct: {result['statistics']['any_correct']}\n\n")

if __name__ == "__main__":
    asyncio.run(main())
