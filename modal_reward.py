import modal

image = (
    modal.Image.debian_slim()
        .pip_install("torch")
        .pip_install("transformers")
        .pip_install("accelerate")
)
app = modal.App("dankreward", image=image)


with image.imports():
    import asyncio
    from typing import List, Dict
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer


# @modal.web_endpoint(method="POST", docs=True)
@app.cls(
    gpu=modal.gpu.L4(),
    # gpu=modal.gpu.T4(),
    # enable_memory_snapshot=True,
    # volumes={"/my_vol": modal.Volume.from_name("my-test-volume")},
    container_idle_timeout=10
)
class Embedder:

    model_id = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    device = "cuda"

    @modal.build()
    def build(self):
        # cache
        print("build")
        dtype = torch.bfloat16
        # dtype = torch.float16
        with torch.device("cuda"):
            model = AutoModelForSequenceClassification.from_pretrained(self.model_id,
                                trust_remote_code=True, torch_dtype=dtype, use_safetensors=True)

    # @modal.enter(snap=True)
    # def load(self):
    #     # Create a memory snapshot with the model loaded in CPU memory.
    #     print("save state")

    # @modal.enter(snap=False)
    @modal.enter()
    def setup(self):
        # Move the model to a GPU before doing any work.
        print("loaded from snapshot")
        dtype = torch.bfloat16
        with torch.device("cuda"):
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id,
                                trust_remote_code=True, torch_dtype=dtype, use_safetensors=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

    # @modal.enter()
    # def setup(self):
    #     # Move the model to a GPU before doing any work.
    #     print("loaded from snapshot")
    #     dtype = torch.float16
    #     self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, device_map="auto", 
    #                            trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True)
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

    # @modal.method()
    @modal.web_endpoint(method="POST", docs=True)
    def score_output(self, messages: List[Dict[str, str]]):
        print("batched")
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to("cuda")
        with torch.no_grad():
            output = self.model(input_ids)
            float_output = output.score.float()
            print("Score:", float_output.item())
        return float_output.item()


# @app.function()
# @modal.web_endpoint(method="POST", docs=True)
# async def run(messages: List[Dict[str, str]]):
#     result = await Embedder().score_output.remote.aio(messages)
#     print(messages, result)
#     return {"result": result}


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