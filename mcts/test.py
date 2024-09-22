import re

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

# Example usage
text1 = """## Step 1: First step
Content of first step.
## Step 2: Second step
Content of second step.
## Step 10: Tenth step
Content of tenth step.
## Step 11: Eleventh step
Content of eleventh step.
sdfsdfsdfsdf



sdfsdfsd

step ## Step 12: Test"""

text2 = """## Step 1: Short step
Brief content.
## Step 99: Large step number
Content of step 99.
## Step 100: Three-digit step
Content of step 100."""

# Test with both examples
for i, text in enumerate([text1, text2], 1):
    # print(f"Test case {i}:")
    result = split_and_clean_steps(text)
    for j, step in enumerate(result, 1):
        print(f"Step {j}:")
        print(step)
        print()
    print("---\n")