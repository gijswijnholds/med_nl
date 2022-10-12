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


all_med_features = ['core args', 'disjunction', 'reverse', 'restrictivity', 'relative clauses', 'quantifiers',
                    'conjunction', 'existential', 'intervals/numbers', 'world knowledge', 'redundancy',
                    'named entities', 'anaphora', 'morphological negation', 'lexical_knowledge', 'conjunction',
                    'common sense', 'conditionals', 'negation', 'npi', 'intersectivity', 'other']

def filter_by_feature(agg_results: List[Tuple[ProcessedSample, List[int]]], feature: str):
    return [sp for sp in agg_results if feature in sp[0].compact.features]

def filter_by_features(agg_results: List[Tuple[ProcessedSample, List[int]]], features: List[str]):
    return [sp for sp in agg_results if any(map(lambda f: f in sp[0].compact.features, features))]


def agg_analysis(agg_results: List[Tuple[ProcessedSample, List[int]]]):
    accuracy_total = compute_accuracy(agg_results)
    accuracy_up = compute_accuracy([sp for sp in agg_results if sp[0].compact.mono == 'UP'])
    accuracy_down = compute_accuracy([sp for sp in agg_results if sp[0].compact.mono == 'DOWN'])
    accuracy_non = compute_accuracy([sp for sp in agg_results if sp[0].compact.mono == 'NON'])
    accuracy_feature = {feature: compute_accuracy(filter_by_feature(agg_results, feature)) for feature in all_med_features}
    return {'total': accuracy_total, 'up': accuracy_up, 'down': accuracy_down, 'non': accuracy_non,
            'feature': accuracy_feature}