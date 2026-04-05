import re
from typing import Optional

def extract_answer(response: str) -> tuple[Optional[str], bool]:
    response = response.strip()

    opens  = re.findall(r"<ANSWER>",  response)
    closes = re.findall(r"</ANSWER>", response)
    is_freq_unique = (len(opens) == 1) and (len(closes) == 1)

    # Properly matched pairs
    matches = re.findall(r"<ANSWER>(.*?)</ANSWER>", response, re.DOTALL)

    is_malformed = not is_freq_unique

    if matches and is_freq_unique:
        return matches[0].strip(), is_malformed

    return None, is_malformed


def extract_reasoning_tag(response: str) -> dict:
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
    if no_tags_at_all:
        is_malformed = False
    else:
        # Other tags
        is_malformed = not (has_match and is_freq_unique)

    return {
        "tag": tag,
        "tags": tags,
        "is_malformed": is_malformed,
    }
