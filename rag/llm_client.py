"""
NeuroRAG — LLM Client Abstraction v2
Unified sync + async interface for local LLaMA and OpenAI.
BaseLLMClient now exposes both complete() (sync) and acomplete() (async).
"""
from __future__ import annotations

import asyncio
import logging
import os
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from configs.settings import LLMConfig, get_config

logger = logging.getLogger(__name__)

_THREAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm-async")


class BaseLLMClient(ABC):
    """
    Abstract LLM client.
    Subclasses must implement complete() (sync).
    acomplete() is provided as a default async wrapper using thread pool.
    """

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str: ...

    async def acomplete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """
        Async wrapper around sync complete().
        Runs in thread pool to avoid blocking the event loop.
        Subclasses may override with a native async implementation.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _THREAD_POOL,
            lambda: self.complete(prompt, system, temperature, max_tokens),
        )


class LocalLLaMAClient(BaseLLMClient):
    """
    Local LLaMA inference via llama-cpp-python.
    Expects LLAMA_MODEL_PATH env var pointing to a .gguf file.
    GPU layers fully offloaded (n_gpu_layers=-1) for RTX 4060.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        from llama_cpp import Llama  # type: ignore[import]
        model_path = os.environ.get("LLAMA_MODEL_PATH", "/models/llama.gguf")
        self._llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False,
        )
        self._cfg = cfg
        logger.info("LocalLLaMAClient: loaded model from %s", model_path)

    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=self._cfg.top_p,
            repeat_penalty=self._cfg.repeat_penalty,
        )
        return response["choices"][0]["message"]["content"]


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client (GPT-4).
    Requires OPENAI_API_KEY env var.
    Provides native async acomplete() via openai's async API.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        import openai  # type: ignore[import]
        self._sync_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._async_client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self._model = cfg.model
        logger.info("OpenAIClient: model=%s", self._model)

    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._sync_client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    async def acomplete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        """Native async — avoids thread pool overhead for OpenAI."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


def build_llm_client(cfg: Optional[LLMConfig] = None) -> BaseLLMClient:
    if cfg is None:
        cfg = get_config().llm
    if cfg.provider == "openai":
        return OpenAIClient(cfg)
    if cfg.provider == "local":
        return LocalLLaMAClient(cfg)
    raise ValueError(f"Unknown LLM provider: {cfg.provider}")


def build_critic_llm_client() -> BaseLLMClient:
    return build_llm_client(get_config().critic_llm)
