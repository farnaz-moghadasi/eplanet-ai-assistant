from typing import Any

import httpx


class ConfluenceClient:
    def __init__(self, base_url: str, cloud_id: str, email: str, api_token: str):
        self.base_url = base_url.rstrip("/")
        self.cloud_id = cloud_id
        self.email = email
        self.api_token = api_token

    @property
    def api_base_url(self) -> str:
        return f"{self.base_url}/ex/confluence/" f"{self.cloud_id}/wiki/api/v2"

    async def get_page(self, page_id: str) -> dict[str, Any]:
        url = f"{self.api_base_url}/pages/{page_id}"

        params = {"body-format": "storage"}

        async with httpx.AsyncClient(
            auth=(self.email, self.api_token), timeout=30.0
        ) as client:
            response = await client.get(
                url, params=params, headers={"Accept": "application/json"}
            )

            response.raise_for_status()

            return response.json()
