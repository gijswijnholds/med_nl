# Preprocess data using a tokenizer etc
from typing import NamedTuple, List
from .datasets import CompactSample, SICK, MED
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class ProcessedSample(NamedTuple):
    premise_tokens: List[int]
    hypothesis_tokens: List[int]
    tokens: List[int]
    compact: CompactSample


def create_tokenizer(tokenizer_name: str):
    return AutoTokenizer.from_pretrained(tokenizer_name)


def tokenize_compact(tokenizer, compact: CompactSample) -> ProcessedSample:
    # TODO: check if we need an EOS token instead of the SEP token
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    prem_tokens = list(map(lambda w: tokenizer.tokenize(w), compact.premise.split()))
    prem_tokens = tokenizer.convert_tokens_to_ids(sum(prem_tokens, []))
    hyp_tokens = list(map(lambda w: tokenizer.tokenize(w), compact.hypothesis.split()))
    hyp_tokens = tokenizer.convert_tokens_to_ids(sum(hyp_tokens, []))
    all_tokens = [cls_token_id] + prem_tokens + [sep_token_id] + hyp_tokens + [sep_token_id]
    return ProcessedSample(prem_tokens, hyp_tokens, all_tokens, compact)


def tokenize_compacts(tokenizer, data: List[CompactSample]) -> List[ProcessedSample]:
    return [tokenize_compact(tokenizer, sample) for sample in data]


class NLIDataset(Dataset):
    def __init__(self, data: List[ProcessedSample]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> ProcessedSample:
        return self.data[i]


def prepare_sick_datasets(sick_path: str, model_name: str) -> List[NLIDataset]:
    sick = SICK(sick_path)
    print("Getting tokenizer...")
    tokenizer = create_tokenizer(model_name)
    print("Tokenizing data...")
    return [NLIDataset(tokenize_compacts(tokenizer, samples))
            for samples in [sick.train_data, sick.dev_data, sick.test_data]]


def prepare_med_datasets(med_path: str, model_name: str) -> List[NLIDataset]:
    med = MED(med_path)
    print("Getting tokenizer...")
    tokenizer = create_tokenizer(model_name)
    print("Tokenizing data...")
    return [NLIDataset(tokenize_compacts(tokenizer, samples))
            for samples in [med.data, med.data, med.data]]


def prepare_datasets(path: str, model_name: str) -> List[NLIDataset]:
    if 'MED' in path:
        return prepare_med_datasets(path, model_name)
    elif 'SICK' in path:
        return prepare_sick_datasets(path, model_name)
    else:
        raise ValueError('Path not valid!')