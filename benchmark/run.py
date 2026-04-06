import json
import asyncio

from typing import Optional, Dict

from llama_client.tokenizer import Tokenizer
from openai import AsyncOpenAI
from .utils import extract_answer, extract_reasoning_tag

from .checker import math_check, exact_match, DatasetEval

ANSWER_CHECKER: Dict[str, DatasetEval] = {
    'commonsense_qa': exact_match,
    'openbook_qa'   : exact_match,
    'gsm8k'         : math_check,
    'math'          : math_check,
    'svamp'         : exact_match,
    'aime2025'      : math_check
}

async def evaluate_single(
    client: AsyncOpenAI,
    tokenizer: Tokenizer,
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
    reasoning_info = extract_reasoning_tag(content)
    answer_info    = extract_answer(content)

    token_usage = tokenizer.count(content)

    if answer_info["is_malformed"]:
        is_correct = False
    else:
        ds_eval = ANSWER_CHECKER.get(sample["dataset_name"], None)

        if ds_eval is None:
            raise ValueError(f"Unrecognized local dataset name : {sample['dataset_name']}")
        
        is_correct = ds_eval(sample["answer"], answer_info["answer"])

    record = {
        "id": sample["index"],
        "token_usage": token_usage,
        "question": sample["question"],
        "ground_truth": sample["answer"],
        "answer_info": answer_info,
        "reasoning_info": reasoning_info,
        "correct": is_correct,
        "raw_response": content,
    }

    if jsonl_file is not None and lock is not None:
        async with lock:
            jsonl_file.write(json.dumps(record) + "\n")
            jsonl_file.flush()

    return idx, record
