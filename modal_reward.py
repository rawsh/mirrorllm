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


@app.cls(
    # gpu=modal.gpu.A10G(),
    gpu=modal.gpu.T4(),
    enable_memory_snapshot=True,
    # volumes={"/my_vol": modal.Volume.from_name("my-test-volume")}
)
class Embedder:

    model_id = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    device = "cuda"

    @modal.build()
    def build(self):
        # cache
        print("build")
        # dtype = torch.bfloat16
        dtype = torch.float16
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id, device_map="auto", 
                               trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        # torch.compile(model)

    @modal.enter(snap=True)
    def load(self):
        # Create a memory snapshot with the model loaded in CPU memory.
        print("save state")
        # dtype = torch.bfloat16
        dtype = torch.float16
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, device_map="cpu", 
                               trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

    @modal.enter(snap=False)
    def setup(self):
        # Move the model to a GPU before doing any work.
        print("loaded from snapshot")
        self.model = self.model.to(self.device)

    @modal.method()
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


@app.function()
@modal.web_endpoint(method="POST", docs=True)
async def run(messages: List[Dict[str, str]]):
    result = await Embedder().score_output.remote.aio(messages)
    print(messages, result)
    return {"result": result}


@app.local_entrypoint()
async def main():
    # score the messages
    prompt = 'What are some synonyms for the word "beautiful"?'
    response1 = 'Nicely, Beautifully, Handsome, Stunning, Wonderful, Gorgeous, Pretty, Stunning, Elegant'
    response2 = 'bad'
    messages1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}]
    messages2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}]
    m1 = Embedder().score_output.remote.aio(messages1)
    m2 = Embedder().score_output.remote.aio(messages2)
    res = await asyncio.gather(*[m1,m2])
    print(response1, res[0])
    print(response2, res[1])