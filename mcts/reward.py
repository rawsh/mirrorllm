from datasets import load_dataset
import numpy as np
from util import split_and_clean_steps, quality_filter, SEED
import json

def initialize_prm(traces, last_step_correct=True):
    """
    Initialize the Process Reward Model (PRM) using sets of reasoning traces.
    
    Args:
    traces (list of list of str): Reasoning traces
    correct (bool): Whether the traces are correct (True) or incorrect (False)
    
    Returns:
    dict: Initialized PRM with quality values and weighted rewards
    """
    # prm = {}
    prm_data = []
    
    for i, trace_tuple in enumerate(traces):
        question, trace = trace_tuple
        K = len(trace)  # Total number of reasoning steps
        
        # Initialize trace
        prm_example = {"steps": [], "quality_values": [], "weighted_rewards": []}
        v_prev = 0
        for k, step in enumerate(trace, 1):
            penalize = (not last_step_correct) and k == len(trace)
            m_k = K - k if (not penalize) else K - k + 1  # One more step needed to correct mistake if incorrect
            r_s_k = 0 if (not penalize) else 1  # 0 for correct steps, 1 for incorrect steps
            w_s_k = (1 - v_prev) / (m_k + 1) * (1 - 2 * r_s_k)
            v_k = max(v_prev + w_s_k, 0)
            
            prm_example["question"] = question
            prm_example["steps"].append(step)
            prm_example["quality_values"].append(v_k)
            prm_example["weighted_rewards"].append(w_s_k)
            v_prev = v_k

        prm_data.append(prm_example)
    
    return prm_data


# Load and filter the dataset, then apply the 90:10 split
ds = load_dataset("argilla/magpie-ultra-v0.1")
# Filter the dataset
filtered_ds = ds.filter(quality_filter)
# Apply the 90:10 split on the filtered training data
split_ds = filtered_ds['train'].train_test_split(test_size=0.1, seed=SEED)
train_ds = split_ds['train']
print(len(train_ds))
# "Correct" traces generated by 405B
correct_traces = [(row["instruction"], split_and_clean_steps(row["response"])) for row in train_ds]

# Example usage:
# correct_traces = [
#     ["Step 1: Correct", "Step 2: Correct", "Step 3: Correct"],
#     ["Step 1: Correct", "Step 2: Correct"]
# ]

with open('out.jsonl') as f:
    last_step_incorrect_data = [json.loads(line) for line in f]
    last_step_incorrect_traces = [(ex["question"], ex["thoughts"]) for ex in last_step_incorrect_data]

# incorrect_traces = [['Identify all the possible outcomes of tossing four coins simultaneously. When tossing four coins simultaneously, each coin has 2 possible outcomes (heads or tails). Therefore, for four coins, the total number of possible outcomes is $2^4 = 16$.', 'List all the outcomes that result in more heads than tails. There are 4 outcomes that meet this criterion: HTHT, HHTT, THTH, TTHH. This gives us a total of 4 favorable outcomes.'], ['Identify all the possible outcomes of tossing four coins simultaneously. When tossing four coins simultaneously, each coin has 2 possible outcomes (heads or tails). Therefore, for four coins, the total number of possible outcomes is $2^4 = 16$.', 'Determine the favorable outcomes. We want more heads than tails, which means we need 3 heads and 1 tail, or 4 heads.', 'Count the number of outcomes with 3 heads and 1 tail. For 3 heads, there is only 1 way to arrange them (HHH). For 1 tail, there are 2 ways to arrange them (TTH and THT). So, there are a total of 1 + 2 = 3 favorable outcomes.'], ['Recognize that this is an arithmetic sequence with a common difference of 1.', 'To find the sum of the first 100 positive integers, we can use the formula for the sum of an arithmetic series, which is given by S = n/2 * (a1 + an), where n is the number of terms, a1 is the first term, and an is the last term.']]

# initialized_prm = initialize_prm(correct_traces)
# print(initialized_prm)
# print(initialized_prm["trace_1000"])

correct_prm_data = initialize_prm(correct_traces, last_step_correct=True)
print(len(correct_prm_data))
total_length = 0
correct_prm_data_step_values = []
for ex in correct_prm_data:
    total_length += len(ex["steps"])
    for i in range(len(ex["steps"])):
        question = ex["question"]
        partial_steps = ex["steps"][:i+1]
        partial_reward = ex["quality_values"][i]
        correct_prm_data_step_values.append({
            "question": question,
            "steps": partial_steps,
            "final_step_reward": partial_reward
        })

print("corr total # step values", total_length)

last_step_incorrect_prm_data = initialize_prm(last_step_incorrect_traces, last_step_correct=False)
print(len(last_step_incorrect_prm_data))

last_step_incorrect_prm_data_step_values = []
for ex in last_step_incorrect_prm_data:
    i = len(ex["steps"]) - 1
    question = ex["question"]
    partial_steps = ex["steps"][:i+1]
    partial_reward = ex["quality_values"][i]
    last_step_incorrect_prm_data_step_values.append({
        "question": question,
        "steps": partial_steps,
        "final_step_reward": partial_reward
    })

print("last step incorr total # step values", len(last_step_incorrect_prm_data_step_values))

# print(initialized_prm)
# print(last_step_incorrect_prm_data[1000])

with open("reward.jsonl", "w") as f:
    for prm_examples in correct_prm_data_step_values:
        json.dump(prm_examples, f)
        f.write("\n")

    for prm_examples in last_step_incorrect_prm_data_step_values:
        json.dump(prm_examples, f)
        f.write("\n")

print(f"Results written to reward.jsonl")