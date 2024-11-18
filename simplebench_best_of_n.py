import aiohttp
import asyncio
import os

MODAL_ENDPOINT = "https://rawsh--reward-api-model-score-dev.modal.run"

async def get_score(session, messages):
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "messages": messages
    }
    
    try:
        async with session.post(MODAL_ENDPOINT, json=payload, headers=headers) as response:
            if response.status != 200:
                text = await response.text()
                print(f"Error {response.status}: {text}")
                print(f"Request payload: {payload}")
                return {"error": text}
            return await response.json()
    except Exception as e:
        print(f"Exception: {str(e)}")
        return {"error": str(e)}

async def main():
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ]
    
    async with aiohttp.ClientSession() as session:
        result = await get_score(session, messages)
        print(result)

if __name__ == "__main__":
    asyncio.run(main())