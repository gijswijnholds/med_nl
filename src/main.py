import os
import torch
from typing import List, Tuple
from transformers import AutoModelForSequenceClassification
from .preprocessing import prepare_datasets
from .trainer import Trainer, Maybe, NLIDataset
from .config import bertje_name, robbert_name, mbert_name, sick_nl_path, med_nl_path
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from .analysis import agg_analysis

GLOBAL_SEEDS = [3, 7, 42]

def get_word_pad_id(bert_name):
    if bert_name == bertje_name: return 3
    elif bert_name == robbert_name: return 1
    elif bert_name == mbert_name: return 0
    else: return None

def setup_trainer(data_path: str,
                  bert_name: str,
                  device: str,
                  seed: int = 42,
                  results_folder: str="./results",
                  model_folder: str="./models") -> Trainer:
    word_pad_id = get_word_pad_id(bert_name)
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


def setup_validator(data_path: str,
                    model_folder: str,
                    bert_name: str,
                    device: str,
                    seed: int = 42) -> Trainer:
    word_pad_id = get_word_pad_id(bert_name)
    torch.manual_seed(seed)
    model_name = f'{bert_name.split("/")[-1]}_{seed}'
    model_path = os.path.join(model_folder, [fn for fn in os.listdir(model_folder) if fn.startswith(model_name)][0])
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    _, val_dataset, test_dataset = prepare_datasets(data_path, bert_name)
    return Trainer(name=model_name,
                   model=model,
                   val_dataset=val_dataset,
                   test_dataset=test_dataset,
                   batch_size_val=128,
                   batch_size_test=128,
                   loss_fn=CrossEntropyLoss(),
                   device=device,
                   word_pad=word_pad_id)

def setup_tester(data_path: str,
                 model_folder: str,
                 bert_name: str,
                 device: str,
                 seed: int = 42) -> Trainer:
    word_pad_id = get_word_pad_id(bert_name)
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


def get_agg_val_results(data_path, model_folder, bert_name, seeds: List[int]) -> float:
    val_accs = []
    for s in seeds:
        trainer = setup_validator(data_path=data_path, model_folder=model_folder,
                                  bert_name=bert_name, device='cpu', seed=s)
        val_loss, val_acc = trainer.eval_epoch(eval_set='val')
        val_accs.append(val_acc)
    return sum(val_accs)/len(val_accs)


def get_agg_test_results(data_path, model_folder, bert_name, seeds: List[int], device: str) -> Tuple[NLIDataset,List[Tuple[int,int,int]]]:
    test_datas = []
    predictionss = []
    for s in seeds:
        trainer = setup_tester(data_path=data_path, model_folder=model_folder,
                               bert_name=bert_name, device=device, seed=s)
        test_datas.append(trainer.test_loader.dataset)
        predictionss.append(trainer.predict_epoch())
    predictionss = [[torch.argmax(p).item() for p in preds] for preds in predictionss]
    return list(zip(test_datas[0], list(zip(*predictionss))))


def main_eval_loop_colab(bert_name):
    sick_test_results = get_agg_test_results(data_path="./drive/MyDrive/data/SICK_NL.txt",
                                             model_folder="./drive/MyDrive/models_sicknl",
                                             bert_name=bert_name,
                                             seeds=[3, 7, 42],
                                             device='cuda')
    sick_analysis = agg_analysis(sick_test_results)
    med_test_results = get_agg_test_results(data_path="./drive/MyDrive/data/MED_NL.tsv",
                                            model_folder="./drive/MyDrive/models_sicknl",
                                            bert_name=bert_name, seeds=[3, 7, 42],
                                            device='cuda')
    med_analysis = agg_analysis(med_test_results)
    return sick_analysis, med_analysis

def main_eval_loop(bert_name, sick_path, med_path, model_folder, device='cpu'):
    sick_test_results = get_agg_test_results(data_path=sick_path,
                                             model_folder=model_folder,
                                             bert_name=bert_name,
                                             seeds=[3, 7, 42],
                                             device=device)
    sick_analysis = agg_analysis(sick_test_results)
    med_test_results = get_agg_test_results(data_path=med_path,
                                            model_folder=model_folder,
                                            bert_name=bert_name, seeds=[3, 7, 42],
                                            device=device)
    med_analysis = agg_analysis(med_test_results)
    return sick_analysis, med_analysis