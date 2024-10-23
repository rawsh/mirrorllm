import modal

image = (
    modal.Image.debian_slim()
        .pip_install("torch")
        .pip_install("transformers")
        .pip_install("accelerate")
)
app = modal.App("mirrorgemma-prm", image=image)


with image.imports():
    from typing import List, Dict, Tuple
    import asyncio
    import torch
    from time import perf_counter as pc
    import copy
    # from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import pipeline
    import os
    # from lib import extract_tensors, test
    # print(test())

@app.cls(
    gpu=modal.gpu.A10G(),
    container_idle_timeout=120,
    # allow_concurrent_inputs=100,
    # volumes={"/data": modal.Volume.from_name("my-test-volume")}
    secrets=[
        modal.Secret.from_name("hf-token"),
        # modal.Secret.from_name("wandb-token")
    ],
)
class Embedder:
    # model_id = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    model_id = "rawsh/mirrorgemma-2-2b-prm-base"
    revision = "a6bc6d57b3d7c873ba30e88ffd3e304e4758c295"
    device = "cuda"
    print(model_id)

    @modal.build()
    def build(self):
        # cache
        print("build")
        dtype = torch.bfloat16
        with torch.device("cuda"):
            # from transformers import (
            #     AutoTokenizer
            # )
            # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", use_auth_token=True)
            # tokenizer.push_to_hub("rawsh/mirrorgemma-2-2b-PRM-base")
            print("[build] loading model")
            start = pc()
            classifier = pipeline("sentiment-analysis", model=self.model_id, revision=self.revision,
                                trust_remote_code=True, torch_dtype=dtype, device="cuda")
            elapsed = pc() - start
            print(f"[build] loading model took {elapsed} seconds")

    # @modal.enter(snap=False)
    @modal.enter()
    def setup(self):
        # Start the model to a GPU before doing any work.
        print("setup")
        # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # faster model loading
        dtype = torch.bfloat16
        with torch.device("cuda"):
            print("[setup] loading model")
            start = pc()
            self.pipeline = pipeline("sentiment-analysis", model=self.model_id, revision=self.revision,
                                trust_remote_code=True, torch_dtype=dtype, device="cuda")
            elapsed = pc() - start
            print(f"[setup] loading model took {elapsed} seconds")

    @modal.web_endpoint(method="POST", docs=True)
    def score_output(self, inp: dict):
        prompt = inp["prompt"]
        print("score_output")
        return self.pipeline(prompt)


# @app.local_entrypoint()
# async def main():
#     # score the messages
#     prompt = 'What are some synonyms for the word "beautiful"?'
#     response1 = 'Nicely, Beautifully, Handsome, Stunning, Wonderful, Gorgeous, Pretty, Stunning, Elegant'
#     response2 = 'bad'
#     messages1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}]
#     messages2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}]
#     m1 = Embedder().score_output(messages1)
#     m2 = Embedder().score_output(messages2)
#     res = await asyncio.gather(*[m1,m2])
#     print(response1, res[0])
#     print(response2, res[1])