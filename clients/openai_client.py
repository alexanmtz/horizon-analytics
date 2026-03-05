import asyncio
from typing import Awaitable, Callable, TypeVar

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError

client = AsyncOpenAI(
    timeout=25.0,
    max_retries=0,
)

T = TypeVar("T")

_TRANSIENT_OPENAI_ERRORS = (
    InternalServerError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)


async def call_openai_with_retries(
    operation: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    base_delay_seconds: float = 0.6,
) -> T | None:
    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except _TRANSIENT_OPENAI_ERRORS:
            if attempt < max_attempts:
                await asyncio.sleep(base_delay_seconds * attempt)
                continue
            return None
        except Exception:
            return None

    return None