import os
import torch
from typing import List, Tuple
from transformers import AutoModelForSequenceClassification
from .preprocessing import prepare_datasets
from .trainer import Trainer, Maybe, NLIDataset
from .config import bertje_name, robbert_name, sick_nl_path, med_nl_path
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from .analysis import agg_analysis

GLOBAL_SEEDS = [3, 7, 42]

# TODO: define trainers and testers
# Train separately from testing
# Need to train on SICK-NL, and on MED-NL (after splitting the models again)
# Split MED by phenomenon to see how generalization is achieved.
# The earlier biased splitting was very complicated and did tone down results but is it worth doing the effort again?


def setup_trainer(data_path: str,
                  bert_name: str,
                  device: str,
                  seed: int = 42,
                  results_folder: str="./results",
                  model_folder: str="./models") -> Trainer:
    word_pad_id = 3 if bert_name == bertje_name else 1 if bert_name == robbert_name else None
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(bert_name, num_labels=2)
    train_dataset, val_dataset, _ = prepare_datasets(data_path, bert_name)
    return Trainer(name=f'{bert_name.split("/")[-1]}_{seed}',
                   model=model,
                   train_dataset=train_dataset,
                   val_dataset=val_dataset,
                   batch_size_train=32,
                   batch_size_val=128,
                   optim_constructor=AdamW,
                   lr=1e-04,
                   loss_fn=CrossEntropyLoss(),
                   device=device,
                   word_pad=word_pad_id,
                   results_folder=results_folder,
                   model_folder=model_folder)

def setup_tester(data_path: str,
                 model_folder: str,
                 bert_name: str,
                 device: str,
                 seed: int = 42) -> Trainer:
    word_pad_id = 3 if bert_name == bertje_name else 1 if bert_name == robbert_name else None
    torch.manual_seed(seed)
    model_name = f'{bert_name.split("/")[-1]}_{seed}'
    model_path = os.path.join(model_folder, [fn for fn in os.listdir(model_folder) if fn.startswith(model_name)][0])
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    _, _, test_dataset = prepare_datasets(data_path, bert_name)
    return Trainer(name=model_name,
                   model=model,
                   test_dataset=test_dataset,
                   batch_size_test=128,
                   device=device,
                   word_pad=word_pad_id)

def train_on_sick():
    trainer = setup_trainer(data_path=sick_nl_path, bert_name=bertje_name, device='cuda', seed=42,
                            results_folder="./drive/MyDrive/results", model_folder="./drive/MyDrive/models")
    trainer.train_loop(num_epochs=20, val_every=1, save_at_best=True)


def test_on_sick():
    trainer = setup_tester(data_path=sick_nl_path, model_folder="./drive/MyDrive/models",
                           bert_name=bertje_name, device='cuda', seed=42)
    return analysis(trainer.test_loader.dataset, trainer.predict_epoch()), trainer.test_loader.dataset

def get_agg_test_results(data_path, model_folder, seeds: List[int]) -> Tuple[NLIDataset,List[Tuple[int,int,int]]]:
    test_datas = []
    predictionss = []
    for s in seeds:
        trainer = setup_tester(data_path=data_path, model_folder=model_folder,
                               bert_name=bertje_name, device='cuda', seed=s)
        test_datas.append(trainer.test_loader.dataset)
        predictionss.append(trainer.predict_epoch())
    predictionss = [[torch.argmax(p).item() for p in preds] for preds in predictionss]
    return list(zip(test_datas[0], list(zip(*predictionss))))


def main_eval_loop():
    sick_test_results = get_agg_test_results(data_path="./drive/MyDrive/data/SICK_NL.txt",
                                             model_folder="./drive/MyDrive/models_sicknl", seeds=[3,7,42])
    sick_analysis = agg_analysis(sick_test_results)
    med_test_results = get_agg_test_results(data_path="./drive/MyDrive/data/MED_NL.tsv",
                                            model_folder="./drive/MyDrive/models_sicknl", seeds=[3, 7, 42])
    med_analysis = agg_analysis(med_test_results)
    return sick_analysis, med_analysis


# sick_test_results = {'total': 0.8739638537844807,
#  'up': -1,
#  'down': -1,
#  'non': 0.8739638537844807,
#  'feature': {'core args': (-1, 0),
#   'disjunction': (-1, 0),
#   'reverse': (-1, 0),
#   'restrictivity': (-1, 0),
#   'relative clauses': (-1, 0),
#   'quantifiers': (-1, 0),
#   'conjunction': (-1, 0),
#   'existential': (-1, 0),
#   'intervals/numbers': (-1, 0),
#   'world knowledge': (-1, 0),
#   'redundancy': (-1, 0),
#   'named entities': (-1, 0),
#   'anaphora': (-1, 0),
#   'morphological negation': (-1, 0),
#   'lexical_knowledge': (-1, 0),
#   'common sense': (-1, 0),
#   'conditionals': (-1, 0),
#   'negation': (-1, 0),
#   'npi': (-1, 0),
#   'intersectivity': (-1, 0),
#   'other': (0.8739638537844807, 4906)}}

# med_test_results = {'total': 0.4755976712498464,
#  'up': 0.6475980931426462,
#  'down': 0.38722493887530507,
#  'non': 0.3949771689497713,
#  'feature': {'core args': (0.0, 2),
#   'disjunction': (0.406824146981627, 254),
#   'reverse': (0.5204918032786885, 244),
#   'restrictivity': (0.5333333333333333, 10),
#   'relative clauses': (0.5, 2),
#   'quantifiers': (0.75, 4),
#   'conjunction': (0.5382803297997644, 283),
#   'existential': (0.75, 4),
#   'intervals/numbers': (1.0, 2),
#   'world knowledge': (0.4999999999999999, 10),
#   'redundancy': (0.8333333333333334, 6),
#   'named entities': (0.5, 2),
#   'anaphora': (1.0, 1),
#   'morphological negation': (0.0, 2),
#   'lexical_knowledge': (0.49524488825487417, 1402),
#   'common sense': (0.6666666666666666, 6),
#   'conditionals': (0.4608501118568231, 149),
#   'negation': (0.5, 2),
#   'npi': (0.36686390532544366, 338),
#   'intersectivity': (0.45833333333333326, 32),
#   'other': (0.47601517179830494, 2988)}}