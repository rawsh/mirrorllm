import modal

image = (
    modal.Image.debian_slim()
        .pip_install("torch")
        .pip_install("transformers")
        .pip_install("accelerate")
)
app = modal.App("dankreward", image=image)


with image.imports():
    from typing import List, Dict, Tuple
    import asyncio
    import torch
    from time import perf_counter as pc
    import copy
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # from lib import extract_tensors, test
    # print(test())

@app.function(
    keep_warm=1
)
@modal.web_endpoint(method="POST")
def upload(model_id):
    def extract_tensors(m: torch.nn.Module) -> Tuple[torch.nn.Module, List[Dict]]:
        """
        Remove the tensors from a PyTorch model, convert them to NumPy
        arrays, and return the stripped model and tensors.
        """
        tensors = []
        for _, module in m.named_modules():
            # Store the tensors in Python dictionaries
            params = {
                name: torch.clone(param).detach().numpy()
                for name, param in module.named_parameters(recurse=False)
            }
            buffers = {
                name: torch.clone(buf).detach().numpy()
                for name, buf in module.named_buffers(recurse=False)
            }
            tensors.append({"params": params, "buffers": buffers})

        # Make a copy of the original model and strip all tensors and
        # temporary buffers out of the copy.
        m_copy = copy.deepcopy(m)
        for _, module in m_copy.named_modules():
            for name in (
                    [name for name, _ in module.named_parameters(recurse=False)]
                    + [name for name, _ in module.named_buffers(recurse=False)]):
                setattr(module, name, None)

        # Make sure the copy is configured for inference.
        m_copy.train(False)
        return m_copy, tensors

    # Create a memory snapshot with the model loaded in CPU memory.
    print("save state")

    # faster model loading
    dtype = torch.float16
    start = pc()
    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, device_map="cpu",
                        trust_remote_code=True, torch_dtype=dtype, use_safetensors=True)
    elapsed = pc() - start
    print(f"loading model on cpu took {elapsed} seconds")
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
    print("extracting tensors")
    m_copy, tensors = extract_tensors(self.model)
    print("save state")

    # faster model loading
    dtype = torch.float16
    start = pc()
    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, device_map="cpu",
                        trust_remote_code=True, torch_dtype=dtype, use_safetensors=True)
    elapsed = pc() - start
    print(f"loading model on cpu took {elapsed} seconds")
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
    print("extracting tensors")
    m_copy, tensors = extract_tensors(self.model)
        for _, module in m_copy.named_modules():
            for name in (
                    [name for name, _ in module.named_parameters(recurse=False)]
                    + [name for name, _ in module.named_buffers(recurse=False)]):
                setattr(module, name, None)

        # Make sure the copy is configured for inference.
        m_copy.train(False)
        return m_copy, tensors

    # Create a memory snapshot with the model loaded in CPU memory.
    print("save state")

    # faster model loading
    dtype = torch.float16
    start = pc()
    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, device_map="cpu",
                        trust_remote_code=True, torch_dtype=dtype, use_safetensors=True)
    elapsed = pc() - start
    print(f"loading model on cpu took {elapsed} seconds")
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
    print("extracting tensors")
    m_copy, tensors = extract_tensors(self.model)
    return True


@app.cls(
    gpu=modal.gpu.L4(),
    container_idle_timeout=10,
    volume=modal.voume("my-test-volume")
)
class Embedder:
    model_id = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    device = "cuda"

    @modal.build()
    def build(self):
        # cache
        print("build")
        dtype = torch.bfloat16
        with torch.device("cpu"):
            model = AutoModelForSequenceClassification.from_pretrained(self.model_id,
                                trust_remote_code=True, torch_dtype=dtype, use_safetensors=True)

    # @modal.enter(snap=True)
    @modal.enter()
    def load(self):
        # move back
        def replace_tensors(m: torch.nn.Module, tensors: List[Dict]):
            """
            Restore the tensors that extract_tensors() stripped out of a 
            PyTorch model.
            :param no_parameters_objects: Skip wrapping tensors in 
            ``torch.nn.Parameters`` objects (~20% speedup, may impact
            some models)
            """
            with torch.device("cuda"):
                modules = [module for _, module in m.named_modules()] 
                for module, tensor_dict in zip(modules, tensors):
                    # There are separate APIs to set parameters and buffers.
                    for name, array in tensor_dict["params"].items():
                        module.register_parameter(name, 
                            torch.nn.Parameter(torch.as_tensor(array)))
                    for name, array in tensor_dict["buffers"].items():
                        module.register_buffer(name, torch.as_tensor(array))  
        
        # Load tensors into the model's graph of Python objects

        # self.model = m_copy
        print("moving mock to cuda")
        start = pc()
        m_copy.to("cuda")
        elapsed = pc() - start
        print(f"moving mock to cuda took {elapsed} seconds")

        print("replacing tensors")
        start = pc()
        replace_tensors(m_copy, tensors)
        elapsed = pc() - start
        print(f"replacing took {elapsed} seconds")
        self.model = m_copy

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}, {"role": "assistant", "content": "wow"}],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to("cuda")
        with torch.no_grad():
            output = self.model(input_ids)
            print(output)

    # @modal.enter(snap=False)
    # @modal.enter()
    # def setup(self):
        # Move the model to a GPU before doing any work.
        # print("loaded from snapshot")
        # self.model.to("cuda")

        # # faster model loading
        # with torch.device("cuda"):
        #     self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id,
        #                         trust_remote_code=True, torch_dtype=dtype, use_safetensors=True)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

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
            print(output)
            float_output = output.score.float()
            print("Score:", float_output.item())
        return float_output.item()


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