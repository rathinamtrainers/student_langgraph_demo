"""LLM integration helper.

Prioritize Gemini (langchain-google-genai) as the default provider when
GOOGLE_API_KEY is available. Falls back to a lightweight MockLLM for offline
or CI/demo runs. This module intentionally does not import Vertex AI — your
manager requested Gemini be preferred for LangChain compatibility.

Usage:
    from research_langgraph_demo.llm import get_llm, HAS_GEMINI
    llm = get_llm()
"""
from typing import Any, List, Optional
import os

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(verbose=True)
except ImportError:
    pass

HAS_GEMINI = False

try:
    # Primary: Gemini via langchain-google-genai
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

    def get_llm(google_api_key: Optional[str] = None, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY not set; cannot create real Gemini LLM instance")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=key)

    HAS_GEMINI = True

except Exception:
    # Fallback mock LLM used for offline/demo runs
    class MockLLM:
        def __init__(self, model: str = "mock", temperature: float = 0.0):
            self.model = model
            self.temperature = temperature

        def chat(self, messages: List[dict]) -> dict:
            # messages: list of {"role": "user|assistant|system", "content": str}
            joined = " ".join(m.get("content", "") for m in messages)
            return {"content": "MOCK_REPLY: " + joined[:200]}

        def generate(self, prompt: str) -> str:
            return "MOCK_GENERATION: " + prompt[:200]

        def bind_tools(self, tools: Any):
            # For compatibility with examples that call llm.bind_tools
            return self

    def get_llm(*args, **kwargs):
        return MockLLM()

def llm_generate(prompt: str) -> str:
    """Generate text with the configured LLM. Works with both Gemini and MockLLM."""
    llm = get_llm()
    if HAS_GEMINI:
        from langchain_core.messages import HumanMessage  # type: ignore
        return llm.invoke([HumanMessage(content=prompt)]).content
    return llm.generate(prompt)


async def llm_agenerate(prompt: str) -> str:
    """Async generate text with the configured LLM."""
    if HAS_GEMINI:
        from langchain_core.messages import HumanMessage  # type: ignore
        llm = get_llm()
        result = await llm.ainvoke([HumanMessage(content=prompt)])
        return result.content
    return get_llm().generate(prompt)


__all__ = ["get_llm", "HAS_GEMINI", "llm_generate", "llm_agenerate"]
