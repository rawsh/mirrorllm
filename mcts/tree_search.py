import asyncio
import math
import aiohttp
import json
from openai import AsyncOpenAI
import time
from functools import wraps
from collections import OrderedDict
from asyncio import Semaphore, TimeoutError

POLICY_URL = 'https://rawsh--vllm-gemma-serve.modal.run/v1/'
PRM_URL = 'https://rawsh--mirrorgemma-prm-embedder-score-output.modal.run'
API_KEY = '9FF74944EED19865193F979942FB1'

POLICY_SEMAPHORE = Semaphore(200)
PRM_SEMAPHORE = Semaphore(1)

MAX_RETRIES = 5
TIMEOUT = 30  # seconds

action_num = 0

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_value = 0
        self.prm_value = None

def async_lru_cache(maxsize=128):
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
            print(f"Attempt {attempt + 1} timed out. Retrying...")
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"Attempt {attempt + 1} failed with error: {str(e)}. Retrying...")
        await asyncio.sleep(1)  # Wait a bit before retrying

async def mcts(root_state, correct_answer, num_iterations, session):
    root = Node(root_state)
    client = AsyncOpenAI(base_url=POLICY_URL, api_key=API_KEY)
    terminal_nodes = set()

    for i in range(num_iterations):
        print(f"Starting iteration {i + 1}/{num_iterations}")
        leaf = select(root)
        is_term, is_corr = await retry_with_timeout(is_terminal, leaf.state, correct_answer, client, session)
        if is_term:
            terminal_nodes.add(leaf.state)
        if not is_term:
            child = await retry_with_timeout(expand, leaf, client, session)
            value = await retry_with_timeout(simulate, child, correct_answer, client, session, terminal_nodes)
            backpropagate(child, value)

    return root, terminal_nodes

def select(node):
    while node.children:
        if len(node.children) < len(get_possible_actions(node.state)):
            return node
        node = best_uct_child(node)
    return node

async def expand(node, client, session):
    action = await retry_with_timeout(get_next_action, node.state, client)
    new_state = apply_action(node.state, action)
    child = Node(new_state, parent=node)
    node.children[action] = child
    return child

async def simulate(node, correct_answer, client, session, terminal_nodes):
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
        depth += 1
    return await retry_with_timeout(evaluate_state, state, session)

def backpropagate(node, value):
    while node:
        node.visits += 1
        node.total_value += value
        node = node.parent

def best_uct_child(node):
    C = 1.41
    return max(
        node.children.values(),
        key=lambda child: (child.total_value / child.visits) + C * math.sqrt(math.log(node.visits) / child.visits)
    )

async def get_next_action(state, client):
    global action_num
    action_num += 1
    print(f"action {action_num}", end="\r")
    prompt = format_state_for_policy(state)
    async with POLICY_SEMAPHORE:
        response = await client.completions.create(
            model="rawsh/mirrorgemma-2-2b-SFT",
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
        print("CORRECT", state)
        return True, True
    
    if state.count("\n\n") < 2:
        return False, False
    
    async with POLICY_SEMAPHORE:
        response = await client.completions.create(
            model="rawsh/mirrorgemma-2-2b-SFT",
            prompt=state,
            max_tokens=1,
            stop=["\n\n"],
            temperature=0.3,
            logprobs=20,
        )
    
    first_token_top_logprobs = response.choices[0].logprobs.top_logprobs[0]
    if "" in first_token_top_logprobs:
        scaled = math.exp(first_token_top_logprobs[""])
        res = response.choices[0].text.strip()

        yes_bigger_than_no = True
        if "\n\n" in first_token_top_logprobs:
            scaled_no = math.exp(first_token_top_logprobs["\n\n"])
            yes_bigger_than_no = (scaled > scaled_no)

        threshold = 0.95
        terminal = (scaled >= threshold) and yes_bigger_than_no
        print(first_token_top_logprobs[""], scaled, res, terminal)
        return terminal, False
    else:
        return False, False

@async_lru_cache(maxsize=1000)
async def evaluate_state(state, session):
    prompt = format_state_for_prm(state)
    async with PRM_SEMAPHORE:
        async with session.post(PRM_URL, json={"prompt": prompt}) as response:
            result = await response.json()
    return float(result[0]['score'])

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

async def run_mcts(initial_state, correct_answer, num_iterations, session):
    start_time = time.time()
    root, terminal_nodes = await mcts(initial_state, correct_answer, num_iterations, session)
    end_time = time.time()
    best_leaf = await find_best_leaf_by_prm(root, session)
    
    terminal_paths = []
    for node in terminal_nodes:
        score = await retry_with_timeout(evaluate_state, node, session)
        terminal_paths.append({
            "final_state": node,
            "score": score,
            "correct": is_correct(node, correct_answer)
        })
    
    result = {
        "question": initial_state,
        "correct_answer": correct_answer,
        "statistics": {
            "num_iterations": num_iterations,
            "execution_time": end_time - start_time,
            "total_terminal_nodes": len(terminal_nodes),
        },
        "best_path": {
            "final_state": best_leaf.state,
            "score": best_leaf.prm_value,
            "correct": is_correct(best_leaf.state, correct_answer)
        },
        "terminal_paths": terminal_paths
    }
    
    return result

async def main():
    initial_states = [
        ("Janet hires six employees. Four of them are warehouse workers who make $15/hour, and the other two are managers who make $20/hour. Janet has to pay 10% of her workers' salaries in FICA taxes. If everyone works 25 days a month and 8 hours a day, how much does Janet owe total for their wages and taxes for one month?", "22000"),
        ("Peggy is moving and is looking to get rid of her record collection. Sammy says that he will buy all of them for 4 dollars each. Bryan is only interested in half of the records but will offer 6 dollars each for the half that he is interested in and 1 dollar each for the remaining half that he is not interested in with the hopes that he can resell them in bulk later. If Peggy has 200 records, what is the difference in profit between Sammy versus Bryan's deal?", "100"),
        ("Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?", "4"),
        ("Carol is an aviation engineer deciding how much fuel to put in a jet. The empty plane needs 20 gallons of fuel per mile. Each person on the plane increases this amount by 3 gallons per mile, and each bag increases it by 2 gallons per mile. If there are 30 passengers and 5 flight crew, and each person brought two bags, how many gallons of fuel does the plane need for a 400-mile trip?", "106000"),
        ("Susan is making jewelry with a repeating pattern that has 3 green beads, 5 purple beads, and twice as many red beads as green beads. If the pattern repeats three times per bracelet and 5 times per necklace, how many beads does she need to make 1 bracelets and 10 necklaces?", "742"),
        ("A group of hawks is called a kettle. It is breeding season for hawks. A group of ornithologists are tracking 6 kettles of hawks. Each kettle has an average of 15 pregnancies that yield 4 babies per batch. How many babies are expected this season if approximately 25% are lost?", "270"),
        ("Brendan makes $6/hour as a waiter. He's scheduled for 2 8-hour shifts and 1 12-hour shift this week. He also makes an average of $12 in tips each hour. Brendan is supposed to pay 20% of his income in taxes, but he only reports 1/3rd of his tips to the IRS. How much money does Brendan pay in taxes each week?", "56"),
        ("Karen's students are about to take a standardized test. Karen gets a $500 bonus if their average score is above 75, plus an extra $10 bonus for every additional point the average score increases above 75. So far, Karen has graded 8 tests, and the average is 70. Given that each student can have a maximum score of 150, what combined score do the last two tests need to have for Karen to earn a $600 bonus?", "290")
    ]
    num_iterations = 10

    async with aiohttp.ClientSession() as session:
        tasks = [run_mcts(state, answer, num_iterations, session) for state, answer in initial_states]
        results = await asyncio.gather(*tasks)
    
    with open("mcts_results.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"\nAll MCTS processes completed and results written to mcts_results.jsonl")

if __name__ == "__main__":
    asyncio.run(main())