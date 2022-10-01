# Define datasets etc.
from typing import NamedTuple

# TODO: turn into an enum (or other) class
UP = 'UP'
DOWN = 'DOWN'
NON = 'NON'


class CompactSample(NamedTuple):
    index: int
    premise: str
    hypothesis: str
    label: int
    mono: str
    features: str


def get_entailment(ent: str) -> str:
    ent_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    return ent_map[ent.lower()]


class SICK(object):
    def __init__(self, sick_fn: str):
        self.sick_fn = sick_fn
        self.name = self.sick_fn.split('/')[-1].split('.')[0]
        self.raw_data = self.load_data()
        self.train_data, self.dev_data, self.test_data, self.data = self.split_data()

    def load_data(self):
        with open(self.sick_fn, 'r') as in_file:
            lines = [ln.strip().split('\t') for ln in in_file.readlines()][1:]
        sentence_data = [tuple(ln[:5]+ln[-1:]) for ln in lines]
        sentence_data = [(int(id), s1 + '.', s2 + '.', el, float(rl), split)
                         for (id, s1, s2, el, rl, split) in sentence_data]
        return sentence_data

    def split_data(self):
        train_data, dev_data, test_data, all_data = [], [], [], []
        for (id, s1, s2, e_label, rl, s) in self.raw_data:
            el = get_entailment(e_label)
            if s == 'TRAIN':
                train_data.append(CompactSample(id, s1, s2, el, NON, []))
                all_data.append(CompactSample(id, s1, s2, el, NON, []))
            if s == 'TRIAL':
                dev_data.append(CompactSample(id, s1, s2, el, NON, []))
                all_data.append(CompactSample(id, s1, s2, el, NON, []))
            if s == 'TEST':
                test_data.append(CompactSample(id, s1, s2, el, NON, []))
                all_data.append(CompactSample(id, s1, s2, el, NON, []))
        return train_data, dev_data, test_data, all_data


def get_monotonicity(props: str) -> str:
    if 'upward_monotone' in props:
        return UP
    elif 'downward_monotone' in props:
        return DOWN
    elif 'non_monotone' in props:
        return NON
    else:
        raise ValueError("No monotonicity found!")


def get_features(props: str) -> list:
    non_features = ['upward_monotone', 'downward_monotone', 'non_monotone']
    return list(filter(lambda p: p not in non_features, props.split(':')))


class MED(object):
    def __init__(self, med_fn: str):
        self.med_fn = med_fn
        self.name = self.med_fn.split('/')[-1].split('.')[0]
        self.data = self.load_data()

    def load_data(self):
        with open(self.med_fn, 'r') as in_file:
            lines = [ln.strip().split('\t') for ln in in_file.readlines()][1:]
        return [CompactSample(int(ln[0]), get_monotonicity(ln[3]), ln[8], ln[9],
                              get_entailment(ln[15]), get_features(ln[3])) for ln in lines]