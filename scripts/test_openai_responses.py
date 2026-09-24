import asyncio

import httpx

from app.core.config import settings


async def main():
    url = "https://api.openai.com/v1/responses"

    async with httpx.AsyncClient(
        timeout=30.0,
    ) as client:
        response = await client.post(
            url,
            headers={
                "Authorization": (
                    f"Bearer {settings.OPENAI_API_KEY}"
                ),
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-5.6-terra",
                "input": "Reply with only: OK",
            },
        )

    print("status:", response.status_code)
    print("body:")
    print(response.text)


if __name__ == "__main__":
    asyncio.run(main())