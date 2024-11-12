import modal

image = (
    modal.Image.debian_slim()
        .pip_install([
            "torch",
            "transformers",
            "accelerate",
            "batched",
        ])
)
# app = modal.App("mirrorqwen-prm", image=image)
app = modal.App("mirrorqwen-prm-st", image=image)

with image.imports():
    from typing import List, Dict, Tuple
    import asyncio
    import torch
    from time import perf_counter as pc
    from transformers import pipeline
    import os

    class BatchProcessor:
        def __init__(self):
            import batched
            self.batched = batched

        def create_batch_processor(self, pipeline_func):
            @self.batched.dynamically(batch_size=256, timeout_ms=200.0, small_batch_threshold=4)
            def _process_batch(prompts: List[str]) -> List[Dict]:
                return pipeline_func(prompts)
            return _process_batch

@app.cls(
    # gpu=modal.gpu.T4(),
    gpu=modal.gpu.A10G(),
    # gpu=modal.gpu.H100(),
    # gpu=modal.gpu.A100(),
    container_idle_timeout=120,
    # allow_concurrent_inputs=1000,
    allow_concurrent_inputs=1000,
    secrets=[
        modal.Secret.from_name("hf-token"),
    ],
)
class Embedder:
    model_id = "rawsh/mirrorqwen2.5-0.5b-prm"
    # revision = "894341fbd81d0c1abdd98b4e0630de932aa63c6f" # base
    revision = "42e07d1b708282ac2aae338050d8116f8c69398d" # st0
    # revision = "65f4a7601dffacc40e0ef7fa4733d346c926bd18" # st1 v1
    # revision = "80da7ccc4f107e0cb6bf937d61be4702badfb96b" # st1 v2
    # revision = "4d618515c90069993f4b32e4201783efdeebbc22" # st2
    # revision = "b052380b619e5c62ce9f407522362f5caf7b8346" # st3
    device = "cuda"
    print(model_id)

    @modal.build()
    def build(self):
        print("build")
        dtype = torch.bfloat16
        with torch.device("cuda"):
            print("[build] loading model")
            start = pc()
            classifier = pipeline("sentiment-analysis", model=self.model_id, revision=self.revision,
                                trust_remote_code=True, torch_dtype=dtype, device="cuda")
            elapsed = pc() - start
            print(f"[build] loading model took {elapsed} seconds")

    @modal.enter()
    def setup(self):
        print("setup")
        dtype = torch.bfloat16
        with torch.device("cuda"):
            print("[setup] loading model")
            start = pc()
            self.pipeline = pipeline("sentiment-analysis", model=self.model_id, revision=self.revision,
                                trust_remote_code=True, torch_dtype=dtype, device="cuda", batch_size=256)
            elapsed = pc() - start
            print(f"[setup] loading model took {elapsed} seconds")
            
            # Initialize batch processor
            batch_processor = BatchProcessor()
            self._process_batch = batch_processor.create_batch_processor(self.pipeline)

    @modal.web_endpoint(method="POST", docs=True)
    async def score_output(self, inp: dict):
        prompt = inp["prompt"]
        # Handle both single inputs and lists of inputs
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        try:
            # Use the batched processing method
            results = await self._process_batch.acall(prompts)
            
            # Return single result if input was single, otherwise return list
            if isinstance(inp["prompt"], str):
                return results[0]
            return results
        except Exception as e:
            return {"error": str(e)}

@app.local_entrypoint()
async def main():
    embedder = Embedder()
    
    # Test with multiple prompts
    prompt = 'What are some synonyms for the word "beautiful"?'
    response1 = 'Nicely, Beautifully, Handsome, Stunning, Wonderful, Gorgeous, Pretty, Stunning, Elegant'
    response2 = 'bad'
    
    # Create batch of requests
    inputs = [
        {"prompt": response1},
        {"prompt": response2}
    ]
    
    # Process in parallel
    results = await asyncio.gather(*[
        embedder.score_output(inp) for inp in inputs
    ])
    
    # Print results
    for response, result in zip([response1, response2], results):
        print(f"Response: {response}\nResult: {result}\n")
    
    # Print batching statistics
    print("Batching stats:", embedder._process_batch.stats)