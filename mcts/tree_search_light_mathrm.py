import asyncio
import math
import aiohttp
from openai import AsyncOpenAI
from datasets import load_dataset
import random
from typing import List, Tuple, Dict, Set, Optional

# Configuration
class Config:
    POLICY_MODEL_NAME = 'MetaMath-Qwen2.5-0.5b'
    POLICY_URL = 'https://rawsh--vllm-qwen-metamath-serve.modal.run/v1/'
    PRM_URL = 'https://rawsh--vllm-qwen-prm-serve.modal.run/v1/'
    PRM_MODEL_NAME = 'MetaMath-Qwen2.5-0.5b-PRM'
    API_KEY = '9FF74944EED19865193F979942FB1'
    UCT_CONSTANT = 1.41
    TIMEOUT = 60
    
class Node:
    def __init__(self, state: str, parent: Optional['Node'] = None):
        self.state = state
        self.parent = parent
        self.children: Dict[str, Node] = {}
        self.visits = 0
        self.total_value = 0.0
        self.prm_value: Optional[float] = None

class MCTSWorker:
    def __init__(self, root_state: str, correct_answer: str):
        self.root = Node(root_state)
        self.correct_answer = correct_answer
        self.terminal_nodes: Set[Node] = set()
        
    async def select_next(self) -> Node:
        """Returns leaf node that should be expanded next."""
        print("\nSELECT_NEXT -----")
        node = self.root
        depth = 0
        path = [node]
        
        while node.children:
            # If node is not fully expanded, return it
            if len(node.children) < 3:  # Assuming 3 possible actions
                print(f"Found unexpanded node at depth {depth}")
                print(f"Current children: {list(node.children.keys())}")
                return node
            node = self._best_uct_child(node)
            path.append(node)
            depth += 1
            
        print(f"Reached leaf node at depth {depth}")
        print(f"Path taken: {' -> '.join(str(n.visits) for n in path)}")
        return node
    
    def _best_uct_child(self, node: Node) -> Node:
        best_value = float('-inf')
        best_child = None
        parent_visits = node.visits
        ln_parent = math.log(parent_visits + 1)
        
        print("\nUCT calculation for children:")
        for action, child in node.children.items():
            if child.visits == 0:
                print(f"Found unvisited child for action: {action}")
                return child
                
            exploit = child.total_value / child.visits
            explore = Config.UCT_CONSTANT * math.sqrt(ln_parent / child.visits)
            uct_value = exploit + explore
            
            print(f"Action: {action[:20]}...")
            print(f"Visits: {child.visits}, Total value: {child.total_value:.3f}")
            print(f"UCT = {exploit:.3f} (exploit) + {explore:.3f} (explore) = {uct_value:.3f}")
            
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
                
        print(f"Selected best child with UCT value: {best_value:.3f}")
        return best_child

class MCTSManager:
    def __init__(self):
        self.policy_client = AsyncOpenAI(
            base_url=Config.POLICY_URL,
            api_key=Config.API_KEY
        )
        self.prm_client = AsyncOpenAI(
            base_url=Config.PRM_URL,
            api_key=Config.API_KEY
        )
        
    async def get_next_action(self, state: str) -> Tuple[str, bool]:
        """Get next action from policy model."""
        print("\nGET_NEXT_ACTION -----")
        print(f"Input state:\n{state}")
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

        try:
            print(f"Calling policy model with {len(messages)} messages")
            response = await self.policy_client.chat.completions.create(
                timeout=Config.TIMEOUT,
                model=Config.POLICY_MODEL_NAME,
                messages=messages,
                max_tokens=150,
                stop=["<|endoftext|>", "<|im_end|>", "\n\n"],
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
            is_term = (response.choices[0].finish_reason == 'stop' and 
                      response.choices[0].stop_reason != '\n\n')
            
            print(f"Generated content: {content}")
            print(f"Is terminal: {is_term}")
            if is_term:
                print(f"TERMINAL STATE REACHED")
                print(f"Q: {question}")
                print(f"A: {content}")
            return content, is_term
            
        except Exception as e:
            print(f"Error getting next action: {e}")
            print(f"Full state that caused error: {state}")
            return "", True

    async def evaluate_state(self, state: str, session: aiohttp.ClientSession) -> float:
        """Evaluate state using PRM model."""
        print("\nEVALUATE_STATE -----")
        print(f"Input state:\n{state}")
        
        steps = state.split("\n\n")
        if len(steps) < 2:
            print("Warning: state has less than 2 steps, returning 0.0")
            return 0.0
            
        question = steps[0]
        curr_step = steps[-1]
        
        messages = []
        if len(steps) == 2:
            messages = [{"role": "user", "content": f"{question} Step 1: {curr_step}"}]
        else:
            messages = [{"role": "user", "content": f"{question} Step 1: {steps[1]}"}]
            for i, step in enumerate(steps[2:-1], start=2):
                messages.extend([
                    {"role": "assistant", "content": "+"},
                    {"role": "user", "content": f"Step {i}: {step}"}
                ])
            messages.extend([
                {"role": "assistant", "content": "+"},
                {"role": "user", "content": f"Step {len(steps)-1}: {curr_step}"}
            ])

        try:
            print(f"Calling PRM model with {len(messages)} messages")
            print(f"Messages: {messages}")
            
            response = await self.prm_client.chat.completions.create(
                timeout=Config.TIMEOUT,
                model=Config.PRM_MODEL_NAME,
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
            print(f"Got logprobs: {logprobs}")
            
            prob_plus = next(
                (math.exp(lp.logprob) for lp in logprobs if lp.token == "+"),
                1e-10
            )
            print(f"Probability for '+' token: {prob_plus}")
            return prob_plus
                
        except Exception as e:
            print(f"Error evaluating state: {e}")
            print(f"Full state that caused error: {state}")
            return 0.0

    async def find_best_leaf(self, node: Node, session: aiohttp.ClientSession) -> Node:
        """Find the best leaf node using PRM values."""
        leaf_nodes: List[Node] = []
        self._collect_leaf_nodes(node, leaf_nodes)
        
        # Evaluate all leaves in parallel
        await asyncio.gather(*[
            self._evaluate_and_update_node(leaf, session) 
            for leaf in leaf_nodes
        ])
        
        return max(leaf_nodes, key=lambda leaf: leaf.prm_value or 0.0)
    
    def _collect_leaf_nodes(self, node: Node, leaf_nodes: List[Node]) -> None:
        """Collect all leaf nodes in the tree."""
        if not node.children:
            leaf_nodes.append(node)
        else:
            for child in node.children.values():
                self._collect_leaf_nodes(child, leaf_nodes)
                
    async def _evaluate_and_update_node(
        self, 
        node: Node, 
        session: aiohttp.ClientSession
    ) -> None:
        """Evaluate a node and update its PRM value."""
        node.prm_value = await self.evaluate_state(node.state, session)

async def run_batch_mcts(
    initial_states: List[Tuple[str, str]], 
    num_trees_per_q: int = 1,
    iters_per_tree: int = 50
) -> List[Dict]:
    """Run batch MCTS on multiple initial states."""
    mcts_manager = MCTSManager()
    
    # Create workers for each question
    all_workers = [
        ([MCTSWorker(state, answer) for _ in range(num_trees_per_q)], answer)
        for state, answer in initial_states
    ]

    async with aiohttp.ClientSession() as session:
        # Run iterations
        for iteration in range(iters_per_tree):
            # Phase 1: Select leaves from all trees
            all_leaves = []
            for workers, _ in all_workers:
                leaves = await asyncio.gather(*[w.select_next() for w in workers])
                all_leaves.extend(leaves)

            # Phase 2: Generate actions for all leaves
            actions = await asyncio.gather(*[
                mcts_manager.get_next_action(leaf.state) for leaf in all_leaves
            ])

            # Phase 3: Apply actions & evaluate
            new_states = [
                (leaf, action, f"{leaf.state}\n\n{action}", is_term) 
                for leaf, (action, is_term) in zip(all_leaves, actions)
            ]
            
            values = await asyncio.gather(*[
                mcts_manager.evaluate_state(state, session) 
                for _, _, state, _ in new_states
            ])

            # Phase 4: Update all trees
            for (leaf, action, new_state, is_term), value in zip(new_states, values):
                if not action:  # Skip empty actions
                    continue
                    
                child = Node(new_state, parent=leaf)
                leaf.children[action] = child
                
                if is_term:
                    for workers, _ in all_workers:
                        for w in workers:
                            if leaf in w.root.children.values():
                                w.terminal_nodes.add(child)
                                
                # Backpropagate
                node = child
                while node:
                    node.visits += 1
                    node.total_value += value
                    node = node.parent

        # Get results
        results = []
        for workers, correct_answer in all_workers:
            best_results = await asyncio.gather(*[
                mcts_manager.find_best_leaf(w.root, session) for w in workers
            ])
            best_leaf = max(best_results, key=lambda x: x.prm_value or 0.0)
            results.append({
                "state": best_leaf.state,
                "score": best_leaf.prm_value,
                "correct": correct_answer in best_leaf.state
            })
    
    return results

async def main():
    # Example dataset loading and processing
    aime = load_dataset("AI-MO/aimo-validation-aime", split="train")
    initial_states = [
        (ex["problem"], str(int(ex["answer"]))) 
        for ex in aime
    ]
    initial_states = random.sample(initial_states, 10)
    
    results = await run_batch_mcts(initial_states)
    accuracy = sum(r['correct'] for r in results) / len(results)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    asyncio.run(main())