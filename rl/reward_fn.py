import re

def extract_answer(text):
    match = re.search(r'<ANSWER>\s*(.+?)\s*</ANSWER>', text)
    return match.group(1) if match else ""

def correctness_reward_func(prompts, completions, answer, **kwargs):
    rewards = []
    print(prompts)
    print(completions)
    print(answer)

    for completion, gold in zip(completions, answer):
        # Extract text from chat format
        response = completion[0]['content']

        gold_str = str(gold).strip()
        ans = extract_answer(response)

        if ans == gold_str:
            rewards.append(1.0)
        else:
            rewards.append(-1.0)

    return rewards