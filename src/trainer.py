import os
import pickle
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch import LongTensor, Tensor, no_grad
from typing import Callable, Any, List, Tuple, Dict
from typing import Optional as Maybe
from .preprocessing import ProcessedSample, NLIDataset
from transfomers import AutoModelForSequenceClassification


def sequence_collator(word_pad: int) -> Callable[[List[ProcessedSample]], Tuple[Tensor, Tensor, Tensor]]:
    def collate_fn(samples: List[ProcessedSample]) -> Tuple[Tensor, Tensor, Tensor]:
        input_ids = pad_sequence([torch.tensor(sample.tokens) for sample in samples],
                                 padding_value=word_pad, batch_first=True)
        input_mask = input_ids != word_pad
        labels = [sample.compact.label for sample in samples]
        return input_ids, input_mask, labels
    return collate_fn


def compute_accuracy(predictions: Tensor, trues: Tensor) -> float:
    # TODO: check whether this still holds for the single label classification model
    return (torch.sum(trues == torch.argmax(predictions, dim=1)) / float(len(predictions))).item()


class Trainer:
    def __init__(self,
                 name: str,
                 model: AutoModelForSequenceClassification,
                 word_pad: int,
                 train_dataset: Maybe[NLIDataset] = None,
                 val_dataset: Maybe[NLIDataset] = None,
                 test_dataset: Maybe[NLIDataset] = None,
                 batch_size_train: Maybe[int] = None,
                 batch_size_val: Maybe[int] = None,
                 batch_size_test: Maybe[int] = None,
                 optim_constructor: Maybe[type] = None,
                 lr: Maybe[float] = None,
                 loss_fn: Maybe[torch.nn.Module] = None,
                 device: str = 'cuda',
                 results_folder: str = 'results'):
        self.name = name
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.device = device
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True,
                                       collate_fn=sequence_collator(word_pad)) if train_dataset else None
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False,
                                     collate_fn=sequence_collator(word_pad)) if val_dataset else None
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False,
                                      collate_fn=sequence_collator(word_pad)) if test_dataset else None
        self.model = model.to(device)
        self.optimizer = optim_constructor(self.model.parameters(), lr=lr) if optim_constructor else None
        self.loss_fn = loss_fn if loss_fn else None

    def save_results(self, results: Dict[int, Dict[str, float]]):
        file_path = f"{self.results_folder}/results_{self.name}.p"
        if os.path.exists(file_path):
            os.remove(file_path)
        with open(file_path, 'wb') as outf:
            pickle.dump(results, outf)

    def train_batch(
            self,
            batch: Tuple[LongTensor, LongTensor, LongTensor]) -> Tuple[float, float]:
        self.model.train()
        input_ids, input_masks, ys = batch
        predictions, _ = self.model.forward(input_ids.to(self.device), input_masks.to(self.device))
        batch_loss = self.loss_fn(predictions, ys.to(self.device))
        accuracy = compute_accuracy(predictions, ys)
        batch_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return batch_loss.item(), accuracy

    def train_epoch(self):
        epoch_loss, epoch_accuracy = 0., 0.
        with tqdm(self.train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                loss, accuracy = self.train_batch(batch)
                tepoch.set_postfix(loss=loss, accuracy=accuracy)
                epoch_loss += loss
                epoch_accuracy += accuracy
        return epoch_loss / len(self.train_loader), epoch_accuracy / len(self.train_loader)

    @no_grad()
    def eval_batch(
            self,
            batch: tuple[LongTensor, LongTensor, LongTensor]) -> Tuple[float, float]:
        self.model.eval()
        input_ids, input_masks, ys = batch
        predictions, _ = self.model.forward(
            input_ids.to(self.device), input_masks.to(self.device))
        batch_loss = self.loss_fn(predictions, ys.to(self.device))
        accuracy = compute_accuracy(predictions, ys)
        return batch_loss.item(), accuracy

    def eval_epoch(self, eval_set: str):
        epoch_loss, epoch_accuracy = 0., 0.
        loader = self.val_loader if eval_set == 'val' else self.test_loader
        batch_counter = 0
        with tqdm(loader, unit="batch") as tepoch:
            for batch in tepoch:
                batch_counter += 1
                loss, accuracy = self.eval_batch(batch)
                tepoch.set_postfix(loss=loss, accuracy=accuracy)
                epoch_loss += loss
                epoch_accuracy += accuracy
        return epoch_loss / len(loader), epoch_accuracy / len(loader)

    @no_grad()
    def predict_batch(
            self,
            batch: Tuple[LongTensor, LongTensor, Any]) -> List[int]:
        self.model.eval()
        input_ids, input_masks, verb_spans, noun_spans, _ = batch
        predictions = self.model.forward(input_ids.to(self.device), input_masks.to(self.device))
        return predictions

    @no_grad()
    def predict_epoch(self) -> List[int]:
        return [label for batch in self.test_loader for label in self.predict_batch(batch)]

    def train_loop(self, num_epochs: int, val_every: int = 1, save_at_best: bool = False):
        results = dict()
        for e in range(num_epochs):
            print(f"Epoch {e}...")
            train_loss, train_acc = self.train_epoch()
            print(f"Train loss {train_loss:.5f}, Train accuracy: {train_acc:.5f}")
            if (e % val_every == 0 and e != 0) or e == num_epochs - 1:
                val_loss, val_acc = self.eval_epoch(eval_set='val')
                print(f"Val loss {val_loss:.5f}, Val accuracy: {val_acc:.5f}")
                if save_at_best and val_acc > max([v['val_acc'] for v in results.values()]):
                    for file in os.listdir('./'):
                        if file.startswith(f'{self.name}'):
                            os.remove(file)
                    self.model.save(f'{self.name}_{e}')
            else:
                val_loss, val_acc = None, -1
            results[e] = {'train_loss': train_loss, 'train_acc': train_acc,
                          'val_loss': val_loss, 'val_acc': val_acc}
            self.save_results(results)
        print(f"Best epoch was {max(results, key=lambda k: results[k]['val_acc'])}")
        return results