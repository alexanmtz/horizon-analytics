from openai import AsyncOpenAI

client = AsyncOpenAI(
    timeout=25.0,
    max_retries=0,
)