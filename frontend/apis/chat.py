import os
from httpx import AsyncClient
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = f"{os.getenv("BACKEND_URL", "http://127.0.0.1:8000")}/chat"


async def send_message_to_chat_api(message: str) -> str:
    async with AsyncClient() as client:
        print(f"sending request to backend {BACKEND_URL}")
        response = await client.post(BACKEND_URL, json={"message": message})
        print(response)
        if response.status_code == 200:
            return response.json()
        else:
            return "Error: Unable to get a response from the chat API."
