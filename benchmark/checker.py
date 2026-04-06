from math_verify import parse, verify, LatexExtractionConfig
from typing import Protocol

class DatasetEval(Protocol):
    def __call__(self, gr_truth: str, predicted: str) -> bool:
        ...


def exact_match(gr_truth: str, predicted: str) -> bool:
    return gr_truth == predicted


def math_check(gr_truth: str, predicted: str) -> bool:
    p_gr_truth = parse(gr_truth, extraction_config=[LatexExtractionConfig()])
    p_predicted = parse(predicted)

    return verify(p_gr_truth, p_predicted)