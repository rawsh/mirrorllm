.



```
curl -X 'POST'   'https://rawsh--vllm-qwen-serve.modal.run/v1/completions'   -H 'accept: application/json'   -H 'Authorization: Bearer 9FF74944EED19865193F979942FB1'   -H 'Content-Type: application/json'   -d '{
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "prompt": [
    "<|im_start|>user\nIf four coins are tossed simultaneously, what is the probability of obtaining more heads than tails, in the coins that land face up? Express your answer as a common fraction.\n<|im_end|>\n<|im_start|>assistant\n## Step 1:\nIdentify all the possible outcomes of tossing four coins simultaneously. When tossing four coins simultaneously, each coin has 2 possible outcomes (heads or tails). Therefore, for four coins, the total number of possible outcomes is $2^4 = 16$.\n\n## Step 2:\n"
  ],
  "max_tokens": 200,
  "stop": ["\n\n## Step "],
  "temperature": 1
}'
```


optimize cot

preference tuning on high quality and low quality chains?
- use tree of thought and track which branches got correct answer
- rate every correct branch using critic model

gen thoughts
- we want high diversity
- possibly off topic or wrong thoughts is fine if we catch them
- signal thought quality based on answer correctness
- no tree of thoughts, just backtracking at inference time?

magpie filter reasoning/math questions
dspy
- question -> chain of thoughts -> answer
- reward based on
- avg quality of thoughts
- quality of answer
- quality of entire output


- optimize just for output quality