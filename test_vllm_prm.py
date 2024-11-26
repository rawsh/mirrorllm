from openai import AsyncOpenAI
import asyncio
import math

class RewardModelClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url="https://rawsh--vllm-qwen-prm-serve.modal.run/v1/",
            api_key="9FF74944EED19865193F979942FB1"
        )
        self.model_name = "MetaMath-Qwen2.5-0.5b-PRM"

    async def get_token_probability(self, response) -> float:
        """Extract probability of + token from response"""
        logprobs = response.choices[0].logprobs.content[0].top_logprobs
        
        # Print tokens and their probabilities for debugging
        token_probs = {lp.token: math.exp(lp.logprob) for lp in logprobs}
        print("Available tokens and probs:", token_probs)
        
        # Get raw probabilities, defaulting to very small number if token not found
        prob_plus = next((math.exp(lp.logprob) for lp in logprobs if lp.token == "+"), 1e-10)
        prob_minus = next((math.exp(lp.logprob) for lp in logprobs if lp.token == "-"), 1e-10)
        
        # Normalize between + and -
        return prob_plus / (prob_plus + prob_minus) if (prob_plus + prob_minus) > 0 else 0.5

    async def evaluate_steps(self, question: str, steps: list[str]) -> list[float]:
        """
        Evaluate each step in the solution getting probabilities of + vs -
        Returns probability of + for each step
        """
        probabilities = []
        
        # First evaluate question + first step
        messages = [
            {"role": "user", "content": f"{question}\n{steps[0]}"}
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=20
            )
            prob = await self.get_token_probability(response)
            probabilities.append(prob)
            
        except Exception as e:
            print(f"Error evaluating first step: {str(e)}")
            probabilities.append(0.5)
        
        # For remaining steps
        for i in range(1, len(steps)):
            try:
                # Build conversation including previous steps
                messages = [
                    {"role": "user", "content": f"{question}\n{steps[0]}"}
                ]
                
                for prev_step in steps[1:i]:
                    messages.extend([
                        {"role": "assistant", "content": "+"},
                        {"role": "user", "content": prev_step}
                    ])
                
                messages.append({"role": "assistant", "content": "+"})
                messages.append({"role": "user", "content": steps[i]})
                
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=1,
                    temperature=0,
                    logprobs=True,
                    top_logprobs=20
                )
                
                prob = await self.get_token_probability(response)
                probabilities.append(prob)
                
            except Exception as e:
                print(f"Error evaluating step {i+1}: {str(e)}")
                probabilities.append(0.5)
            
        return probabilities

async def main():
    # Initialize client
    reward_model = RewardModelClient()

    # Example problem
    question = "Janet has 3 apples and buys 2 more. How many apples does she have?"
    steps = [
        "Step 1: If Janet has 3 apples and buys 2 more, total apples = 3 + 2 = 5.",
        "Step 2: Therefore, Janet has 5 apples. The answer is: 5",
    ]

    try:
        # Get evaluations
        probabilities = await reward_model.evaluate_steps(question, steps)
        
        # Print results
        print("\nResults:")
        print("Question:", question)
        print("\nStep Evaluations:")
        for step, prob in zip(steps, probabilities):
            print(f"P(+) = {prob:.3f}: {step}")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())