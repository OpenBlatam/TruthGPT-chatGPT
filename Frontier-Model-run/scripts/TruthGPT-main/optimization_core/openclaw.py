"""
OpenClaw Compatibility Layer for TruthGPT.
Allows users to 'import openclaw' and use the same high-level API.
Integrates directly with the OpenClaw Deep Refiner V2 Gateway (System 5.9).
"""

import aiohttp
import asyncio
import time
from typing import Optional

from truthgpt import api as _api
from agents.client import AgentClient
from agents.models import AgentConfig, AgentResponse

# Alias for the main API instance
api = _api

# Re-export key methods for direct access if needed
ask = _api.ask
list_papers = _api.list_papers
get_paper_info = _api.get_paper_info
apply_paper = _api.apply_paper

# Gateway Client for OpenClaw Deep Refiner V2
OPENCLAW_GATEWAY_URL = "http://127.0.0.1:18789"

async def deep_refine(prompt: str, hours: float = 0.016, criteria: str = "Clarity, impact, and fidelity to the prompt", provider: str = "deepseek") -> Optional[str]:
    """
    Sends a refinement task to the local OpenClaw Deep Refiner Gateway (System 5.9).
    This function acts as an asynchronous client that polls for the result.
    """
    async with aiohttp.ClientSession() as session:
        # 1. Submit the job
        payload = {
            "prompt": prompt,
            "hours": hours,
            "criteria": criteria,
            "provider": provider,
            "branches": 2,
            "top_k": 2
        }
        try:
            async with session.post(f"{OPENCLAW_GATEWAY_URL}/refine", json=payload) as response:
                if response.status != 202:
                    print(f"[OpenClaw] Gateway error: {response.status}")
                    return None
                
                data = await response.json()
                job_id = data.get("job_id")
                print(f"[OpenClaw] Task submitted to Deep Refiner. Job ID: {job_id}")
        except Exception as e:
            print(f"[OpenClaw] Connection to Gateway failed. Is 'claw --serve' running? Error: {e}")
            return None

        # 2. Poll for completion
        while True:
            await asyncio.sleep(5)
            try:
                async with session.get(f"{OPENCLAW_GATEWAY_URL}/jobs/{job_id}") as poll_res:
                    if poll_res.status == 200:
                        status_data = await poll_res.json()
                        if status_data["status"] == "completed":
                            print(f"[OpenClaw] Refinement complete! Score: {status_data.get('score')}")
                            return status_data.get("output")
                        elif status_data["status"] == "failed":
                            print("[OpenClaw] Refinement failed.")
                            return None
            except Exception as e:
                print(f"[OpenClaw] Polling error: {e}")
                return None

__all__ = ["api", "ask", "list_papers", "get_paper_info", "apply_paper", "AgentClient", "AgentConfig", "AgentResponse", "deep_refine"]

