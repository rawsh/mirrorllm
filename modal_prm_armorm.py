from typing import List, Dict
import modal

image = modal.Image.debian_slim().pip_install([
    "torch", "transformers", "accelerate", "batched", "hf_transfer"
]).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

app = modal.App("reward-api", image=image)

MODEL_NAME = "RLHFlow/ArmoRM-Llama3-8B-v0.1"

with image.imports():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from batched import inference

    def validate_messages(messages: List[Dict[str, str]]):
        if not messages or len(messages) < 2:
            raise ValueError("Messages must contain at least a user and assistant message")
        if not all(isinstance(m, dict) and 'role' in m and 'content' in m for m in messages):
            raise ValueError("Each message must have 'role' and 'content' fields")
            
    class RewardModelHelper:
        def __init__(self, model):
            self.model = model
            
        @inference.dynamically(batch_size=64, timeout_ms=20.0)
        def score_batch(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
            with torch.no_grad():
                # Move input to same device as model
                inputs = {k: v.to(self.model.device) for k, v in features.items()}
                return self.model(inputs["input_ids"]).score.float()

@app.cls(
    gpu=modal.gpu.A10G(),
    allow_concurrent_inputs=1000,
    container_idle_timeout=300,
)
class Model:
    def load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        )
        return model

    @modal.build()
    def build(self):
        self.load_model()

    @modal.enter()
    def setup(self):
        self.model = self.load_model()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True,
        )
        self.score_batch = RewardModelHelper(self.model).score_batch

    @modal.web_endpoint(method="POST")
    async def score(self, messages_dict: Dict[str, List[Dict[str, str]]]):
        messages = messages_dict["messages"]
        validate_messages(messages)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            tokenize=True
        )
        score = await self.score_batch.acall({"input_ids": inputs})
        return {"score": score[0].item()}