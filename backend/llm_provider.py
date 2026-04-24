from groq import AsyncGroq
from config import GROQ_API_KEY, GROQ_MODEL

_client = AsyncGroq(api_key=GROQ_API_KEY)


async def call_llm(prompt: str) -> str:
    response = await _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
