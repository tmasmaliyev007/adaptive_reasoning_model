import json
import asyncio

from pathlib import Path
from tqdm.asyncio import tqdm

from typing import Optional
from datasets import load_dataset

from .run import evaluate_single

from llama_client.tokenizer import Tokenizer
from openai import AsyncOpenAI

async def evaluate_dataset(
    client: AsyncOpenAI,
    tokenizer: Tokenizer,
    filepath: str,
    limit: Optional[int] = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 1.0,
    repeat_penalty: float = 1.0,
    output_dir: Optional[str] = None,
    concurrency: int = 10,
) -> dict:
    dataset = load_dataset('json', data_files={'test': filepath})['test']
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    jsonl_file = None
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        jsonl_file = open(out_path / "results.jsonl", "w", encoding="utf-8")

    tasks = [
        evaluate_single(
            client=client,
            tokenizer = tokenizer,
            semaphore=semaphore,
            idx=i,
            sample=sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            jsonl_file=jsonl_file,
            lock=lock,
        )
        for i, sample in enumerate(dataset)
    ]

    correct = 0
    unparseable = 0
    token_count = 0
    results = []

    with tqdm(total=len(tasks), desc="Dataset") as pbar:
        for coro in asyncio.as_completed(tasks):
            idx, record = await coro
            results.append((idx, record))

            if record["answer_info"]["is_malformed"]:
                unparseable += 1
            elif record["correct"]:
                correct += 1

            token_count+= record["token_usage"]
            done = len(results)

            pbar.set_postfix(
                acc = f"{correct / done:.1%}", 
                avg_token_count = f"{token_count / done:.1f}",
                correct = correct, 
                unparseable = unparseable
            )
            pbar.update(1)

    if jsonl_file is not None:
        jsonl_file.close()

    total = len(dataset)
    accuracy = correct / total if total > 0 else 0.0
    avg_token_count = token_count / total if total > 0 else 0.0

    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "average_token_count": avg_token_count,
        "total": total,
        "unparseable": unparseable,
    }

    if output_dir is not None:
        with open(Path(output_dir) / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary