import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate

from dotenv import load_dotenv
load_dotenv()
    
task_model = dspy.OpenAI(model="gpt-4o-mini", max_tokens=4000)

dspy.settings.configure(lm=task_model)