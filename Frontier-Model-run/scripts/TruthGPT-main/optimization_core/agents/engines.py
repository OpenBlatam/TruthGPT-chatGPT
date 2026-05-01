"""
Centralized Inference Engine Registry and Resilience Layer for TruthGPT.
"""

import logging
import asyncio
from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable
from tenacity import retry, wait_exponential, stop_after_attempt
from .models import InferenceResult

logger = logging.getLogger(__name__)

@runtime_checkable
class AsyncLLMEngine(Protocol):
    """Standard protocol for an asynchronous LLM inference engine."""
    async def __call__(self, prompt: str) -> Union[str, InferenceResult]:
        ...

class DummyAsyncLLM:
    """Mock engine that echos back information about the prompt."""
    async def __call__(self, prompt: str) -> str:
        return f"Echo from OpenClaw Agent (Mock): Prompt length {len(prompt)}."

import httpx
class DeepSeekAsyncLLM:
    """DeepSeek API Engine for TruthGPT."""
    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "sk-27ad7c86391441528616a78ae6eb09cf")
        self.url = "https://api.deepseek.com/chat/completions"

    async def __call__(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Intentar primero con el modelo más potente para código
        models = ["deepseek-reasoner", "deepseek-chat"]
        
        last_err = None
        for model in models:
            data = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1 if model == "deepseek-chat" else None # Reasoner no usa temp
            }
            # Timeout masivo para permitir razonamiento profundo
            async with httpx.AsyncClient(timeout=180.0) as client:
                try:
                    logger.info(f"Attempting DeepSeek synthesis with model: {model}")
                    response = await client.post(self.url, headers=headers, json=data)
                    if response.status_code == 200:
                        result = response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        err_msg = f"API Error {response.status_code}: {response.text}"
                        logger.error(err_msg)
                        with open("synthesis_error.log", "a") as f:
                            f.write(f"{time.ctime()} - {model} - {err_msg}\n")
                        last_err = f"API Error {response.status_code}"
                except Exception as e:
                    err_msg = f"Network/Timeout Error: {e}"
                    logger.error(err_msg)
                    with open("synthesis_error.log", "a") as f:
                        f.write(f"{time.ctime()} - {model} - {err_msg}\n")
                    last_err = str(e)
                    
        raise Exception(f"All DeepSeek models failed: {last_err}")

def resilient_llm_call(func):
    """Decorator to add exponential backoff to LLM calls."""
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True
    )
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    return wrapper

class EngineRegistry:
    """Registry to manage and switch between different LLM engines."""
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EngineRegistry, cls).__new__(cls)
            cls._instance._engines = {}
            cls._instance.register("mock", DummyAsyncLLM())
            cls._instance.register("deepseek", DeepSeekAsyncLLM())
        return cls._instance

    def register(self, name: str, engine: AsyncLLMEngine):
        """Register a new LLM engine instance."""
        self._engines[name] = engine
        logger.info(f"Engine Registry: Engine '{name}' registered.")

    def get_engine(self, name: str) -> Optional[AsyncLLMEngine]:
        """Retrieve an engine by its registered name."""
        return self._engines.get(name)

    def get_all_engines(self) -> Dict[str, AsyncLLMEngine]:
        """Return all registered engines."""
        return self._engines.copy()

# Global singleton
engine_registry = EngineRegistry()

async def safe_llm_call(engine: AsyncLLMEngine, prompt: str, trace_id: Optional[str] = None) -> str:
    """
    Executes an LLM call with retry logic and telemetry.
    """
    from .observability import global_tracer
    
    span = global_tracer.start_span(
        trace_id or "default", 
        name="llm_inference", 
        kind="llm_call", 
        input_data=prompt[-500:]
    )
    
    try:
        # Tenacity retry is handled by the calling layer or can be wrapped here
        @resilient_llm_call
        async def _call():
            return await engine(prompt)
            
        res = await _call()
        res_text = res.text if hasattr(res, 'text') else str(res)
        
        # Capture token usage if available in metadata
        tokens = {}
        if hasattr(res, 'metadata') and res.metadata:
            tokens = {k: v for k, v in res.metadata.items() if "token" in k.lower()}
            
        span.finish(output=res_text, metadata=tokens)
        return res_text
    except Exception as e:
        logger.error(f"LLM Call failed after retries: {e}")
        span.finish(output=str(e), status="error")
        raise

