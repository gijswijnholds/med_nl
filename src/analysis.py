from typing import List, Tuple
from .preprocessing import NLIDataset, ProcessedSample

def analysis(test_data: NLIDataset, predictions: List[int]):
    pass


def compute_accuracy(agg_results: List[Tuple[ProcessedSample, List[int]]]) -> float:
    def single_accuracy(sample_preds: Tuple[ProcessedSample, List[int]]) -> float:
        return sum([p == sample_preds[0].compact.label for p in sample_preds[1]]) / len(sample_preds[1])
    if agg_results:
        return sum(map(single_accuracy, agg_results)) / len(agg_results)
    else:
        return -1


def agg_analysis(agg_results: List[Tuple[ProcessedSample, List[int]]]):
    accuracy_total = compute_accuracy(agg_results)
    accuracy_up = compute_accuracy([sp for sp in agg_results if sp[0].compact.mono == 'UP'])
    accuracy_down = compute_accuracy([sp for sp in agg_results if sp[0].compact.mono == 'DOWN'])
    accuracy_non = compute_accuracy([sp for sp in agg_results if sp[0].compact.mono == 'NON'])
    return {'total': accuracy_total, 'up': accuracy_up, 'down': accuracy_down, 'non': accuracy_non}