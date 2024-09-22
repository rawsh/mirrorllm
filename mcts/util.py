
import re

SEED = 42

def split_and_clean_steps(text):
    # Use regex to split the text into steps
    steps = re.split(r'(?=##\s*Step\s+\d+:)', text)
    
    # Remove any leading/trailing whitespace, empty steps, and the "## Step n:" prefix
    cleaned_steps = []
    for step in steps:
        # Strip whitespace and check if step is not empty
        step = step.strip()
        if step:
            # Remove the "## Step n:" prefix
            step = re.sub(r'^##\s*Step\s+\d+:\s*', '', step)
            cleaned_steps.append(step)
    
    return cleaned_steps

def quality_filter(example):
    response_quality = example['score'] >= 0.32 # arbitrary af 
    # TODO: check correctness of chain
    # math_and_reasoning = example['primary_tag'] in ['Math', 'Reasoning']
    instruction_quality = example['quality'] in ['excellent', 'good']
    response_format = "## Step 1: " in example['response']
    return response_quality and instruction_quality and response_format