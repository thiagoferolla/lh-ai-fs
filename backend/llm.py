import os
from typing import TypeVar

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


def is_llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def _ensure_llm_available() -> None:
    if not is_llm_available():
        raise LLMError("OPENAI_API_KEY is not configured")


def call_llm(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0,
) -> str:
    """Call the LLM through LangChain and return the response content."""
    _ensure_llm_available()
    llm = ChatOpenAI(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    response = llm.invoke(messages)
    content = response.content
    if not isinstance(content, str):
        return str(content)
    return content


def call_llm_json(
    messages: list[dict],
    schema: type[T],
    model: str | None = None,
    temperature: float = 0,
) -> T:
    """Call the LLM through LangChain and return validated structured output."""
    _ensure_llm_available()
    if not issubclass(schema, BaseModel):
        raise TypeError("schema must be a Pydantic BaseModel subclass")
    llm = ChatOpenAI(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    try:
        return llm.with_structured_output(schema).invoke(messages)
    except Exception as exc:  # noqa: BLE001 - surface validation failures as agent errors.
        raise LLMError(f"LLM structured output failed: {exc}") from exc
