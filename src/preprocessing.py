# Preprocess data using a tokenizer etc
from typing import NamedTuple
from .datasets import CompactSample, SICK, MED
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class ProcessedSample(NamedTuple):
    premise_tokens: list[int]
    hypothesis_tokens: list[int]
    tokens: list[int]
    sample: CompactSample


def create_tokenizer(tokenizer_name: str):
    return AutoTokenizer.from_pretrained(tokenizer_name)


def tokenize_compact(tokenizer, compact: CompactSample) -> ProcessedSample:
    # TODO: check if we need an EOS token instead of the SEP token
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    prem_tokens = list(map(lambda w: tokenizer.tokenize(w), compact.premise.split()))
    prem_tokens = tokenizer.convert_tokens_to_ids(sum(prem_tokens, []))
    hyp_tokens = list(map(lambda w: tokenizer.tokenize(w), compact.hypothesis.split()))
    hyp_tokens = tokenizer.convert_tokens_to_ids(sum(prem_tokens, []))
    all_tokens = [cls_token_id] + prem_tokens + [sep_token_id] + hyp_tokens + [sep_token_id]
    return ProcessedSample(prem_tokens, hyp_tokens, all_tokens, compact)


def tokenize_compacts(tokenizer, data: list[CompactSample]) -> list[ProcessedSample]:
    return [tokenize_compact(tokenizer, sample) for sample in data]


class NLIDataset(Dataset):
    def __init__(self, data: list[ProcessedSample]):
        self.data = data
        for sample in self.data:
            sample.check()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> ProcessedSample:
        return self.data[i]


def prepare_sick_datasets(sick_path: str, model_name: str) -> list[NLIDataset]:
    sick = SICK(sick_path)
    print("Getting tokenizer...")
    tokenizer = create_tokenizer(model_name)
    print("Tokenizing data...")
    return [NLIDataset(tokenize_compacts(tokenizer, samples))
            for samples in [sick.train_data, sick.dev_data, sick.test_data]]


def prepare_med_datasets(med_path: str, model_name: str) -> list[NLIDataset]:
    med = MED(med_path)
    print("Getting tokenizer...")
    tokenizer = create_tokenizer(model_name)
    print("Tokenizing data...")
    return [NLIDataset(tokenize_compacts(tokenizer, samples))
            for samples in [med.train_data, med.dev_data, med.test_data]]


def prepare_datasets(path: str, model_name: str) -> list[NLIDataset]:
    if 'MED' in path:
        return prepare_med_datasets(path, model_name)
    elif 'SICK' in path:
        return prepare_sick_datasets(path, model_name)
    else:
        raise ValueError('Path not valid!')