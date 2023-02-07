from typing import List, Tuple
from .preprocessing import NLIDataset, ProcessedSample

relevant_features = ['lexical_knowledge', 'disjunction', 'conjunction', 'conditionals', 'npi', 'reverse']


def compute_accuracy(agg_results: List[Tuple[ProcessedSample, List[int]]]) -> float:
    def single_accuracy(sample_preds: Tuple[ProcessedSample, List[int]]) -> float:
        return sum([p == sample_preds[0].compact.label for p in sample_preds[1]]) / len(sample_preds[1])

    if agg_results:
        return round(100 * (sum(map(single_accuracy, agg_results)) / len(agg_results)), 2)
    else:
        return -1


def filter_by_feature(agg_results: List[Tuple[ProcessedSample, List[int]]], feature: str):
    return [sp for sp in agg_results if feature in sp[0].compact.features]


def compute_feature_accuracies(agg_results: List[Tuple[ProcessedSample, List[int]]]):
    accuracy_feature = {feature: (compute_accuracy(filter_by_feature(agg_results, feature)),
                                  len(filter_by_feature(agg_results, feature))) for feature in relevant_features}
    other_results = [sp for sp in agg_results if not any(map(lambda f: f in sp[0].compact.features, relevant_features))]
    accuracy_feature['other'] = (compute_accuracy(other_results), len(other_results))
    return accuracy_feature


def agg_analysis(agg_results: List[Tuple[ProcessedSample, List[int]]]):
    accuracy_total = compute_accuracy(agg_results)
    up_results = [sp for sp in agg_results if sp[0].compact.mono == 'UP']
    down_results = [sp for sp in agg_results if sp[0].compact.mono == 'DOWN']
    non_results = [sp for sp in agg_results if sp[0].compact.mono == 'NON']
    accuracy_up = compute_accuracy(up_results)
    accuracy_down = compute_accuracy(down_results)
    accuracy_non = compute_accuracy(non_results)
    accuracy_feature = compute_feature_accuracies(agg_results)
    accuracy_up_feature = compute_feature_accuracies(up_results)
    accuracy_down_feature = compute_feature_accuracies(down_results)
    accuracy_non_feature = compute_feature_accuracies(non_results)
    return {'total': accuracy_total, 'up': accuracy_up, 'down': accuracy_down, 'non': accuracy_non,
            'total_feature': accuracy_feature, 'up_feature': accuracy_up_feature,
            'down_feature': accuracy_down_feature, 'non_feature': accuracy_non_feature}