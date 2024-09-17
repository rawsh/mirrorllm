.

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