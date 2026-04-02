import re
import json

from pathlib import Path
from tqdm.auto import tqdm

from typing import Optional
from datasets import load_dataset, concatenate_datasets

from openai import OpenAI

def _build_prompt(question: str, choices: dict) -> str:
    options = " ".join(
        f"({label}){text}"
        for label, text in zip(choices["label"], choices["text"])
    )
    return (
        f"{question}\n"
        f"{options}"
    )


def _extract_answer(response: str) -> tuple[Optional[str], bool]:
    response = response.strip()

    opens  = re.findall(r"<ANSWER>",  response)
    closes = re.findall(r"</ANSWER>", response)
    is_freq_unique = (len(opens) == 1) and (len(closes) == 1)

    # Properly matched pairs
    matches = re.findall(r"<ANSWER>(.*?)</ANSWER>", response, re.DOTALL)
    has_match = len(matches) == 1

    is_malformed = not (has_match and is_freq_unique)

    if matches and not is_malformed:
        return matches[0].strip(), is_malformed

    return None, is_malformed


def _extract_reasoning_tag(response: str) -> dict:
    response = response.strip()
    TAG = r"(LONG_COT|COT|CODE)"

    opens  = re.findall(rf"<{TAG}>",  response)
    closes = re.findall(rf"</{TAG}>", response)
    is_freq_unique = (len(opens) == 1) and (len(closes) == 1)

    # Properly matched pairs
    matched = re.findall(rf"<{TAG}>(.*?)</\1>", response, re.DOTALL)
    has_match = len(matched) == 1

    no_tags_at_all = not opens and not closes

    tags = [tag for tag, _ in matched]
    tag = tags[0] if tags else ("DIRECT" if no_tags_at_all else None)

    # If direct Answer
    if not no_tags_at_all:
        is_malformed = False
    else:
        # Other tags
        is_malformed = not (has_match and is_freq_unique)

    return {
        "tag": tag,
        "tags": tags,
        "is_malformed": is_malformed,
    }


def evaluate_commonsense_qa(
    client: OpenAI,
    limit: Optional[int] = None,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 1.0,
    repeat_penalty: float = 1.0,
    output_dir: Optional[str] = None,
) -> dict:
    splits = ["train", "validation", "test"]
    dataset = concatenate_datasets([
        load_dataset("tau/commonsense_qa", split=s) for s in splits
    ])
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    correct = 0
    unparseable = 0

    jsonl_file = None
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        jsonl_file = open(out_path / "results.jsonl", "w", encoding="utf-8")

    pbar = tqdm(enumerate(dataset), total=len(dataset), desc="CommonsenseQA")
    for i, sample in pbar:
        question = sample["question"]
        choices = sample["choices"]
        answer_key = sample["answerKey"]

        prompt = _build_prompt(question, choices)
        message = [{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            messages = message,
            model="any",
            temperature = temperature,
            top_p = top_p,
            max_completion_tokens = max_new_tokens,
            extra_body = {
                "repeat_penalty": repeat_penalty,
                "top_k": top_k
            }
        )
        print(response)
        response = response.choices[0].message.content

        predicted, answer_malformed = _extract_answer(response)
        reasoning = _extract_reasoning_tag(response)
        is_correct = predicted == answer_key

        if predicted is None:
            unparseable += 1
        elif is_correct:
            correct += 1

        record = {
            "id": sample["id"],
            "question": question,
            "answer_key": answer_key,
            "predicted": predicted,
            "answer_malformed": answer_malformed,
            "correct": is_correct,
            "reasoning_tag": reasoning["tag"],
            "reasoning_all_tags": reasoning["tags"],
            "reasoning_malformed": reasoning["is_malformed"],
            "raw_response": response,
        }

        if jsonl_file is not None:
            jsonl_file.write(json.dumps(record) + "\n")
            jsonl_file.flush()


        acc_so_far = correct / (i + 1)
        pbar.set_postfix(correct=f"{correct}/{i+1}", acc=f"{acc_so_far:.3f}")

    if jsonl_file is not None:
        jsonl_file.close()

    total = len(dataset)
    accuracy = correct / total if total > 0 else 0.0

    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "unparseable": unparseable,
    }

    if output_dir is not None:
        with open(Path(output_dir) / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary