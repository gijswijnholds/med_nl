from typing import List, Tuple
from src.preprocessing import NLIDataset, ProcessedSample
import pickle


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


def compute_overlap_feature_accuracies(results_model1: List[Tuple[ProcessedSample, List[int]]],
                                       results_model2: List[Tuple[ProcessedSample, List[int]]]):
    accuracy_feature = {feature: (compute_overlap_accuracy(filter_by_feature(results_model1, feature),
                                                           filter_by_feature(results_model2, feature)),
                                  len(filter_by_feature(results_model1, feature))) for feature in relevant_features}
    other_results1 = [sp for sp in results_model1 if
                      not any(map(lambda f: f in sp[0].compact.features, relevant_features))]
    other_results2 = [sp for sp in results_model2 if
                      not any(map(lambda f: f in sp[0].compact.features, relevant_features))]
    accuracy_feature['other'] = (compute_overlap_accuracy(other_results1, other_results2), len(other_results1))
    return accuracy_feature


def compute_overlap_accuracy(results_model1: List[Tuple[ProcessedSample, List[int]]],
                             results_model2: List[Tuple[ProcessedSample, List[int]]]):
    overlap = [d for i, d in enumerate(results_model1) if results_model2[i][1] == d[1]]
    non_overlap_model1 = [d for i, d in enumerate(results_model1) if results_model2[i][1] != d[1]]
    non_overlap_model2 = [results_model2[i] for i, d in enumerate(results_model1) if results_model2[i][1] != d[1]]
    overlap_percentage = round(100 * len(overlap) / len(results_model1), 2) if results_model1 else -1
    return {'overlap_perc': overlap_percentage,
            'overlap_acc': compute_accuracy(overlap),
            'model1_acc': compute_accuracy(non_overlap_model1),
            'model2_acc': compute_accuracy(non_overlap_model2)}


def overlap_analysis(results_model1: List[Tuple[ProcessedSample, List[int]]],
                     results_model2: List[Tuple[ProcessedSample, List[int]]]):
    accuracy_total_model1 = compute_accuracy(results_model1)
    accuracy_total_model2 = compute_accuracy(results_model2)
    accuracy_total = compute_overlap_accuracy(results_model1, results_model2)
    up_results1, up_results2 = [sp for sp in results_model1 if sp[0].compact.mono == 'UP'], [sp for sp in results_model2
                                                                                             if
                                                                                             sp[0].compact.mono == 'UP']
    down_results1, down_results2 = [sp for sp in results_model1 if sp[0].compact.mono == 'DOWN'], [sp for sp in
                                                                                                   results_model2 if sp[
                                                                                                       0].compact.mono == 'DOWN']
    non_results1, non_results2 = [sp for sp in results_model1 if sp[0].compact.mono == 'NON'], [sp for sp in
                                                                                                results_model2 if sp[
                                                                                                    0].compact.mono == 'NON']
    accuracy_up = compute_overlap_accuracy(up_results1, up_results2)
    accuracy_down = compute_overlap_accuracy(down_results1, down_results2)
    accuracy_non = compute_overlap_accuracy(non_results1, non_results2)
    accuracy_feature = compute_overlap_feature_accuracies(results_model1, results_model2)
    accuracy_up_feature = compute_overlap_feature_accuracies(up_results1, up_results2)
    accuracy_down_feature = compute_overlap_feature_accuracies(down_results1, down_results2)
    accuracy_non_feature = compute_overlap_feature_accuracies(non_results1, non_results2)
    return {'total_acc_model1': accuracy_total_model1,
            'total_acc_model2': accuracy_total_model2,
            'total': accuracy_total, 'up': accuracy_up, 'down': accuracy_down, 'non': accuracy_non,
            'total_feature': accuracy_feature, 'up_feature': accuracy_up_feature,
            'down_feature': accuracy_down_feature, 'non_feature': accuracy_non_feature}


all_features = relevant_features + ['other']

def average_dict_vals(dicts: List[dict]):
  def safe_avg(vals):
      safe_vals = [v for v in vals if v>=0.]
      return round(sum(safe_vals) / len(safe_vals), 2) if safe_vals else -1
  return {k: safe_avg([local_dict[k] for local_dict in dicts]) for k in dicts[0]}

def average_list_vals(vals: List):
  return sum(vals)/len(vals)

def average_overlap_analysis(model1_resultss: List[List[Tuple[ProcessedSample, List[int]]]],
                             model2_resultss: List[List[Tuple[ProcessedSample, List[int]]]]):
  overlap_analyses = [overlap_analysis(results1, results2) for results1 in model1_resultss for results2 in model2_resultss]
  no_analyses = len(overlap_analyses)
  average_analysis = {'total_acc_model1': average_list_vals([a['total_acc_model1'] for a in overlap_analyses]),
                      'total_acc_model2': average_list_vals([a['total_acc_model2'] for a in overlap_analyses]),
                      'total': average_dict_vals([a['total'] for a in overlap_analyses]),
                      'up': average_dict_vals([a['up'] for a in overlap_analyses]),
                      'down': average_dict_vals([a['down'] for a in overlap_analyses]),
                      'non': average_dict_vals([a['non'] for a in overlap_analyses])}
  average_analysis['total_feature'] = {f: average_dict_vals([a['total_feature'][f][0] for a in overlap_analyses]) for f in all_features}
  average_analysis['up_feature'] = {f: average_dict_vals([a['up_feature'][f][0] for a in overlap_analyses]) for f in all_features}
  average_analysis['down_feature'] = {f: average_dict_vals([a['down_feature'][f][0] for a in overlap_analyses]) for f in all_features}
  average_analysis['non_feature'] = {f: average_dict_vals([a['non_feature'][f][0] for a in overlap_analyses]) for f in all_features}
  return average_analysis



def main_overlap_analysis(med_test_results_model1, med_test_results_model2):
    med_test_results_model11 = [(d[0], [d[1][0]]) for d in med_test_results_model1]
    med_test_results_model12 = [(d[0], [d[1][1]]) for d in med_test_results_model1]
    med_test_results_model13 = [(d[0], [d[1][2]]) for d in med_test_results_model1]

    med_test_results_model21 = [(d[0], [d[1][0]]) for d in med_test_results_model2]
    med_test_results_model22 = [(d[0], [d[1][1]]) for d in med_test_results_model2]
    med_test_results_model23 = [(d[0], [d[1][2]]) for d in med_test_results_model2]
    return average_overlap_analysis(
        [med_test_results_model11, med_test_results_model12, med_test_results_model13],
        [med_test_results_model21, med_test_results_model22, med_test_results_model23])


def write_agreements_to_latex_text(feature: str,
                                   overlap_percentage: float,
                                   overlap_accuracy: float,
                                   model1_accuracy: float,
                                   model2_accuracy: float) -> str:
  def fix_feature(f):
    if f == 'npi': return "NPI"
    elif f == 'lexical_knowledge': return "Lexical"
    else: return f.capitalize()
  return f"\t& {{\\small \\emph{{{fix_feature(feature)}}}}} & {round(overlap_percentage)}\\% & {overlap_accuracy} & {model1_accuracy} & {model2_accuracy} \\\\"

def write_agreements_to_latex_text_features(analysis):
  all_agreements = [write_agreements_to_latex_text(f, analysis[f]['overlap_perc'],
                                              analysis[f]['overlap_acc'], analysis[f]['model1_acc'], analysis[f]['model2_acc']) for f in analysis]
  return "\n".join(all_agreements)


def get_print_agreements(model1_results_fn, model2_results_fn):
    with open(model1_results_fn, 'rb') as inf1:
        med_test_results_model1 = pickle.load(inf1)[1]
    with open(model2_results_fn, 'rb') as inf2:
        med_test_results_model2 = pickle.load(inf2)[1]
    avg_overlap_analysis = main_overlap_analysis(med_test_results_model1, med_test_results_model2)
    print(write_agreements_to_latex_text_features(avg_overlap_analysis['total_feature']))
    print("\\midrule")
    print(write_agreements_to_latex_text_features(avg_overlap_analysis['up_feature']))
    print("\\midrule")
    print(write_agreements_to_latex_text_features(avg_overlap_analysis['down_feature']))
    print("\\midrule")
    print(write_agreements_to_latex_text_features(avg_overlap_analysis['non_feature']))