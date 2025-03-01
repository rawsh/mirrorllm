import modal
import asyncio
from contextlib import asynccontextmanager

def download_model_to_image(model_dir, model_name, model_revision):
    import os
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)
    snapshot_download(
        model_name,
        revision=model_revision,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
    )
    move_cache()

MODEL_DIR = "/qwen"
MODEL_NAME = "rawsh/MetaMath-Qwen2.5-0.5b"
MODEL_REVISION = "286ca8b160074c923b89c318652ab4b979627550"
# MODEL_NAME = "rawsh/mirrorqwen2.5-0.5b-ORPO-3"
# MODEL_REVISION = "4b3e3eb18fe84477ee949058484ec951a5b8beb6"

vllm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.6.2",
        "torch==2.4.0",
        "transformers>=4.45",
        "ray==2.36.0",
        "hf-transfer==0.1.8",
        "huggingface_hub==0.25.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        secrets=[modal.Secret.from_name("hf-token")],
        kwargs={
            "model_dir": MODEL_DIR,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
        },
    )
    .env({"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"})
)

app = modal.App("vllm-qwen-metamath")

N_GPU = 1
MINUTES = 60
HOURS = 60 * MINUTES

async def get_model_config(engine):
    try:
        return await engine.get_model_config()
    except Exception as e:
        print(f"Error getting model config: {e}")
        raise

@asynccontextmanager
async def lifespan(app):
    # Startup
    try:
        await asyncio.sleep(0)  # Give chance for event loop to start
        yield
    finally:
        # Shutdown: Cancel all pending tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

@app.function(
    image=vllm_image,
    gpu=modal.gpu.A10G(count=N_GPU),
    container_idle_timeout=2 * MINUTES,
    timeout=20 * MINUTES,
    allow_concurrent_inputs=1000,
    secrets=[modal.Secret.from_name("vllm-token")]
)
@modal.asgi_app()
def serve():
    import os
    import fastapi
    import vllm.entrypoints.openai.api_server as api_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.logger import RequestLogger
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
    from vllm.entrypoints.openai.serving_engine import BaseModelPath
    from vllm.usage.usage_lib import UsageContext

    web_app = fastapi.FastAPI(
        title=f"OpenAI-compatible {MODEL_NAME} server",
        description="Run an OpenAI-compatible LLM server with vLLM on modal.com",
        version="0.0.1",
        docs_url="/docs",
        lifespan=lifespan
    )

    http_bearer = fastapi.security.HTTPBearer(
        scheme_name="Bearer Token",
        description="See code for authentication details.",
    )
    web_app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    TOKEN = os.environ["API_TOKEN"]
    async def is_authenticated(api_key: str = fastapi.Security(http_bearer)):
        if api_key.credentials != TOKEN:
            raise fastapi.HTTPException(
                status_code=fastapi.status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
            )
        return {"username": "authenticated_user"}

    router = fastapi.APIRouter(dependencies=[fastapi.Depends(is_authenticated)])
    
    # wrap vllm's router in auth router
    router.include_router(api_server.router)
    # add authed vllm to our fastAPI app
    web_app.include_router(router)

    engine_args = AsyncEngineArgs(
        model=MODEL_DIR,
        tensor_parallel_size=N_GPU,
        gpu_memory_utilization=0.90,
        max_model_len=8096,
        enforce_eager=False,
        enable_prefix_caching=True
    )

    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    async def setup_engine():
        model_config = await get_model_config(engine)
        return model_config

    # Use asyncio.run to properly handle the async setup
    model_config = asyncio.run(setup_engine())
    request_logger = RequestLogger(max_log_len=2048)

    base_model_paths = [
        BaseModelPath(name=MODEL_NAME.split("/")[1], model_path=MODEL_NAME)
    ]
    
    # Set up completion endpoint
    api_server.completion = lambda s: OpenAIServingCompletion(
        engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
    )

    # Set up chat endpoint
    api_server.chat = lambda s: OpenAIServingChat(
        engine,
        model_config=model_config,
        base_model_paths=base_model_paths,
        lora_modules=[],
        prompt_adapters=[],
        request_logger=request_logger,
        response_role="assistant"
    )

    return web_app