import re
from typing import Optional, Dict

def detect_tag(tags: Dict[str, str], response: str) -> Optional[str]:
    occurance = None
    for e_tag, e_pattern in tags.items():
        matches = re.search(e_pattern, response, flags=re.DOTALL)

        if matches:
            occurance = e_tag
            break
        
    return occurance

def extract_answer(response: str) -> tuple[Optional[str], bool]:
    response = response.strip()

    TAGS = {
        "ANSWER" : r"<ANSWER>(.*)</ANSWER>", 
    }

    SINGLE_TAGS = {
        "ANSWER_OPEN"        : r"<ANSWER>",
        "ANSWER_CLOSED"      : r"</ANSWER>",
    }

    answer                  = None
    inner_answer_tag        = None
    inner_single_answer_tag = None

    outer_answer_tag        = None
    outer_single_answer_tag = None

    is_malformed            = False


    for tag, pattern in TAGS.items():
        mathces = re.search(pattern, response, flags=re.DOTALL)
        if mathces is None:
            continue
        
        # Get Inner text
        group = mathces.group(1)

        # Check if there is a open & closed tag inside the selected text
        inner_answer_tag = detect_tag(TAGS, group)
        if inner_answer_tag:
            break

        # Check if there is a single tag inside the selected text
        inner_single_answer_tag = detect_tag(SINGLE_TAGS, group)
        if inner_single_answer_tag:
            break
        
        st_l = mathces.start(0)
        st_r = mathces.start(1)

        end_l = mathces.end(1)
        end_r = mathces.end(0)

        outer_group = response[0: st_l] + response[st_r: end_l] + response[end_r: ]

        # Check if there is a open & closed tag outside the selected text
        outer_answer_tag = detect_tag(TAGS, outer_group)
        if outer_answer_tag:
            break

        # Check if there is a single tag outside the selected text
        outer_single_answer_tag = detect_tag(SINGLE_TAGS, outer_group)
        if outer_single_answer_tag:
            break
        
        # If all conditions are met, hence, it is applicable to select final answer
        answer = group.strip()
        break

    if (inner_answer_tag or \
        inner_single_answer_tag or \
        outer_answer_tag or \
        outer_single_answer_tag or \
        answer is None
    ):
        is_malformed = True

    return {
        "answer"                  : answer,
        "inner_answer_tag"        : inner_answer_tag,
        "inner_single_answer_tag" : inner_single_answer_tag,
        "outer_answer_tag"        : outer_answer_tag,
        "outer_single_answer_tag" : outer_single_answer_tag,
        "is_malformed"            : is_malformed
    }


def extract_reasoning_tag(response: str) -> dict:
    response = response.strip()
    TAGS = {
        "LONG_COT" : r"<LONG_COT>(.*)</LONG_COT>", 
        "COT"      : r"<COT>(.*)</COT>",
        "CODE"     : r"<CODE>(.*)</CODE>"
    }

    SINGLE_TAGS = {
        "COT_OPEN"        : r"<COT>",
        "COT_CLOSED"      : r"</COT>",
        "LONG_COT_OPEN"   : r"<LONG_COT>",
        "LONG_COT_CLOSED" : r"</LONG_COT>",
        "CODE_OPEN"       : r"<CODE>",
        "CODE_CLOSED"     : r"</CODE>"
    }

    reason_tag              = None
    inner_reason_tag        = None
    inner_single_reason_tag = None

    outer_reason_tag        = None
    outer_single_reason_tag = None

    is_malformed            = False

    for tag, pattern in TAGS.items():
        mathces = re.search(pattern, response, flags=re.DOTALL)
        if mathces is None:
            continue
        
        # Get Inner text
        group = mathces.group(1)

        # Check if there is a open & closed tag inside the selected text
        inner_reason_tag = detect_tag(TAGS, group)
        if inner_reason_tag:
            break

        # Check if there is a single tag inside the selected text
        inner_single_reason_tag = detect_tag(SINGLE_TAGS, group)
        if inner_single_reason_tag:
            break
        
        st_l = mathces.start(0)
        st_r = mathces.start(1)

        end_l = mathces.end(1)
        end_r = mathces.end(0)

        outer_group = response[0: st_l] + response[st_r: end_l] + response[end_r: ]

        # Check if there is a open & closed tag outside the selected text
        outer_reason_tag = detect_tag(TAGS, outer_group)
        if outer_reason_tag:
            break

        # Check if there is a single tag outside the selected text
        outer_single_reason_tag = detect_tag(SINGLE_TAGS, outer_group)
        if outer_single_reason_tag:
            break
        
        # If all conditions are met, hence, it is applicable to select tag as task-tag
        reason_tag = tag
        break

    if (inner_reason_tag or inner_single_reason_tag or outer_reason_tag or outer_single_reason_tag):
        is_malformed = True
    else:
        reason_tag = "DIRECT" if reason_tag is None else reason_tag

    return {
        "tag"                     : reason_tag,
        "inner_reason_tag"        : inner_reason_tag,
        "inner_single_reason_tag" : inner_single_reason_tag,
        "outer_reason_tag"        : outer_reason_tag,
        "outer_single_reason_tag" : outer_single_reason_tag,
        "is_malformed"            : is_malformed
    }