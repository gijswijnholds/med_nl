import os
import torch
from typing import List, Tuple
from transformers import AutoModelForSequenceClassification
from .preprocessing import prepare_datasets
from .trainer import Trainer, Maybe, NLIDataset
from .config import bertje_name, robbert_name, sick_nl_path, med_nl_path
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from .analysis import analysis

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