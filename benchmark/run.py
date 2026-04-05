import json
import asyncio

from typing import Optional

from openai import AsyncOpenAI
from .utils import extract_answer, extract_reasoning_tag

async def evaluate_single(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    idx: int,
    sample: dict,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repeat_penalty: float,
    jsonl_file=None,
    lock: Optional[asyncio.Lock] = None,
) -> tuple[int, dict]:
    prompt = sample["question"]
    message = [{"role": "user", "content": prompt}]

    async with semaphore:
        response = await client.chat.completions.create(
            messages=message,
            model="any",
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_new_tokens,
            extra_body={
                "repeat_penalty": repeat_penalty,
                "top_k": top_k,
            }
        )

    content = response.choices[0].message.content
    predicted, answer_malformed = extract_answer(content)
    reasoning = extract_reasoning_tag(content)

    record = {
        "id": sample["index"],
        "question": sample["question"],
        "answer": sample["answer"],
        "predicted": predicted,
        "answer_malformed": answer_malformed,
        "correct": predicted == sample["answer"],
        "reasoning_tag": reasoning["tag"],
        "reasoning_all_tags": reasoning["tags"],
        "reasoning_malformed": reasoning["is_malformed"],
        "raw_response": content,
    }

    if jsonl_file is not None and lock is not None:
        async with lock:
            jsonl_file.write(json.dumps(record) + "\n")
            jsonl_file.flush()

    return idx, record
