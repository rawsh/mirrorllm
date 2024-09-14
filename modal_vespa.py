import modal


# vespa_image = modal.Image.from_registry("vespaengine/vespa", add_python="3.11")
vespa_image = modal.Image.from_dockerfile("Dockerfile", add_python="3.11")
app = modal.App("dankvespa", image=vespa_image)

@modal.web_endpoint(method="POST", docs=True)
@app.cls(
    enable_memory_snapshot=True,
    # volumes={"/my_vol": modal.Volume.from_name("my-test-volume")}
)
class Vespa:
    @modal.build()
    def build(self):
        # cache
        print("build")

    @modal.enter(snap=True)
    def load(self):
        # Create a memory snapshot with the model loaded in CPU memory.
        print("save state")

    @modal.enter(snap=False)
    def setup(self):
        # Move the model to a GPU before doing any work.
        print("loaded from snapshot")

    @modal.method()
    def search(self, query: str):
        print("search")


@app.local_entrypoint()
async def main():
    # score the messages
    m1 = await Vespa().search.remote.aio("test")
    print(m1)