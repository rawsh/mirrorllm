from typing import List, Dict, Tuple
import asyncio
import torch
import copy
from transformers import AutoModelForSequenceClassification, AutoTokenizer

dtype = torch.bfloat16
with torch.device("cuda"):
    model = AutoModelForSequenceClassification.from_pretrained(self.model_id,
                        trust_remote_code=True, torch_dtype=dtype, use_safetensors=True)
