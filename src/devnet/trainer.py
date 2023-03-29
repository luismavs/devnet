from __future__ import annotations

import os
import sys
import time
from enum import Enum, auto
from logging import Logger, getLogger
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import TrainerConfig


class Model(nn.Module):
    def __init__(self, n_input: int, inner_neurons :int = 20, state_dict=None):
        """Define a Devnet model

            #TODO: option to add extra hidden laters

        Args:
            n_input: input dimension
            inner_neurons: hidden layer dimensions. Defaults to 20.
            state_dict: to switch between train and predict modes. Defaults to None.
        """        

        super(Model, self).__init__()
        self.n_input = n_input
        self.fc = nn.Linear(n_input, inner_neurons)
        self.output = nn.Linear(inner_neurons, 1)

        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.output.weight, gain=nn.init.calculate_gain("relu"))
        if state_dict is not None:
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.output(x)


class Phase(Enum):
    TRAIN = auto()
    EVAL = auto()
    PREDICT = auto()


class BaseDataset(Dataset):
    LABEL_NONE = -1

    def __init__(self, phase: Phase) -> None:
        self.phase = phase

    def setup(self, x, y=None):
        self.xs = torch.tensor(x, dtype=torch.float32)

        if y is None:
            self.ys = torch.tensor([self.LABEL_NONE], dtype=torch.int8).expand(len(self.xs))
        else:
            self.ys = torch.tensor(y, dtype=torch.int8)

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]

    def __len__(self):
        return len(self.ys)


class TableDataset(BaseDataset):
    LABEL_NAME = "class"

    def __init__(self, df: pd.DataFrame, phase: Phase, logger: Logger) -> None:
        super().__init__(phase)
        self._column_names = df.columns
        self.logger = logger

        if self.LABEL_NAME not in df.columns and phase != Phase.PREDICT:
            raise ValueError(f"columns of '{self.LABEL_NAME}' is necesasry on train or eval phase")

        if self.LABEL_NAME not in df.columns:
            self.setup(df.values)
        else:
            self.setup(df.drop(columns=[self.LABEL_NAME]).values, df.loc[:, self.LABEL_NAME].values)

        self.logger.info("==============================")
        self.logger.info(f"TableDataset {phase.name}")
        self.logger.info(f"data shape: {self.xs.shape}")
        self.logger.info(f"n_inliner: {torch.sum(self.ys == 0)}")
        self.logger.info(f"n_outliner: {torch.sum(self.ys == 1)}")
        self.logger.info("==============================")

    @property
    def n_columns(self):
        return self.xs.shape[1]

    @property
    def df(self):
        tensor = self.xs
        if self.ys[0] != self.LABEL_NONE:
            tensor = torch.cat([self.xs, self.ys.unsqueeze(1)], dim=1)
        df = pd.DataFrame(tensor.numpy())
        df.columns = self._column_names
        return df


class BalancedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self, dataset: BaseDataset, n_batch: int, batch_size: int, seed: int
    ) -> None:
        self.dataset = dataset
        self.n_batch = n_batch
        self.batch_size = batch_size

        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

        self.n_samples_per_class = batch_size // 2
        self.inlier_indices = (dataset.ys == 0).nonzero().squeeze()
        self.outlier_indices = (dataset.ys == 1).nonzero().squeeze()

    def __iter__(self):
        for _ in range(self.n_batch):
            yield self._choice(self.inlier_indices) + self._choice(self.outlier_indices)

    def __len__(self) -> int:
        return self.batch_size

    def _choice(self, data: torch.Tensor) -> list[int]:
        if len(data) == 0:
            return []

        indices = torch.randint(
            high=len(data), size=(self.n_samples_per_class,), generator=self.gen
        )
        return data[indices].tolist()


class Trainer:
    def __init__(self, config: TrainerConfig, logger: Optional[Logger] = None) -> None:
        self.config = config
        self.logger = getLogger(__name__) if logger is None else logger

        torch.manual_seed(config.random_seed)
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(config.model_path), exist_ok=True)

        self._model: Optional[Model] = None
        self._optimizer: Optional[optim.Optimizer] = None
        self.dataloader: dict[Phase, DataLoader] = {}

    @property
    def model(self) -> Model:
        if self._model is None:
            raise ValueError("model is not initialized")
        return self._model

    @property
    def optimizer(self) -> optim.Optimizer:
        if self._optimizer is None:
            raise ValueError("optimizer is not initialized")
        return self._optimizer

    def _setup(self, phase: Phase, df=None):
        if phase in self.dataloader:
            return

        if df is None:
            path = os.path.join(self.config.dataroot, f"{phase.name.lower()}.csv")
            if phase == Phase.PREDICT and self.config.predict_input:
                path = self.config.predict_input

            self.logger.info(f"phase: {phase}, load data from {path}")
            df = pd.read_csv(path)

        dataset = TableDataset(df, phase, self.logger)
        self.dataloader[phase] = self.create_dataloader(dataset)

        if self._model is None:
            loaded_data = None
            if os.path.isfile(self.config.model_path):
                self.logger.info(f"load model from {self.config.model_path}")
                loaded_data = torch.load(
                    self.config.model_path, map_location=self.config.device
                )

            if phase == Phase.PREDICT and loaded_data is None:
                raise ValueError("model_path is necessary on predict")

            self._model = Model(
                n_input = dataset.n_columns if loaded_data is None else loaded_data["n_input"],
                state_dict = None if loaded_data is None else loaded_data["model"],
            )

        assert (
            self.model.n_input == dataset.n_columns
        ), "model.n_input should be same with n_column of dataset"

        if phase == Phase.TRAIN:
            self._optimizer = optim.RMSprop(
                self.model.parameters(), lr=0.001, alpha=0.9, eps=1e-7, weight_decay=0.01
            )

    def create_dataloader(self, dataset: BaseDataset) -> DataLoader:
        if dataset.phase != Phase.TRAIN:
            return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        sampler = BalancedBatchSampler(
            dataset,
            self.config.n_batch,
            self.config.batch_size,
            self.config.random_seed,
        )
        return torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

    def save(self, model_path: str) -> None:
        data = {
            "model": self.model.state_dict(),
            "n_input": self.model.n_input,
        }

        torch.save(data, model_path)
        self.logger.info(f"save model to {model_path}")

    def forward(self, x, y):
        y_pred = self.model(x).squeeze()

        ref = torch.normal(mean=0, std=1, size=(5000,))
        score = (y_pred - torch.mean(ref)) / torch.std(ref)

        inlier = (1 - y) * torch.abs(score)
        outlier = y * torch.maximum(torch.zeros_like(score), 5 - score)

        return torch.mean(inlier + outlier)

    def _train(self, dataloader: DataLoader) -> float:
        losses = []

        for i, (x, y) in enumerate(dataloader):
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            
            loss = self.forward(x, y)
            loss.backward()
            losses.append(loss.item())

            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return sum(losses) / len(losses)

    def train(self, df_train: pd.DataFrame = None, df_eval: pd.DataFrame = None) -> None:
        self._setup(Phase.TRAIN, df=df_train)
        self.model.train()

        for epoch in range(self.config.epochs):
            start_time = time.time()
            loss = self._train(self.dataloader[Phase.TRAIN])

            if epoch % self.config.log_interval == 0:
                elapsed_time = time.time() - start_time
                self.logger.info(
                    "[train] epoch: {}, loss: {:.2f}, time: {:.2f}".format(
                        epoch, loss, elapsed_time
                    )
                )

            if epoch % self.config.eval_interval == 0:
                self.eval(epoch, df=df_eval)

        self.eval(-1, is_report=True)

    @torch.no_grad()
    def eval(self, epoch: int, is_report: bool = False, is_save: bool = True, df: pd.DataFrame = None) -> None:
        self._setup(Phase.EVAL, df=df)
        self.model.eval()

        y_preds, y_trues = self.predict_scores(self.dataloader[Phase.EVAL])

        roc_auc = metrics.roc_auc_score(y_trues, y_preds)
        ap = metrics.average_precision_score(y_trues, y_preds)
        self.logger.info(
            f"[eval] epoch: {epoch}, AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap)
        )

        if is_save:
            self.save(self.config.model_path)

        if is_report:
            self.report(y_trues, y_preds)

    @torch.no_grad()
    def predict(self, df_predict: pd.DataFrame = None) -> pd.DataFrame:
        self._setup(Phase.PREDICT, df_predict)
        self.model.eval()

        dataloader = self.dataloader[Phase.PREDICT]
        y_preds, _ = self.predict_scores(dataloader)

        df = dataloader.dataset.df
        df["score"] = y_preds

        config = self.config
        if config.predict_output:
            os.makedirs(os.path.dirname(config.predict_output), exist_ok=True)
            df.to_csv(config.predict_output)
            self.logger.info(f"write predict result to {config.predict_output}")

        return df

    @torch.no_grad()
    def predict_scores(self, dataloader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        y_preds = []
        y_trues = []
        for i, (x, y) in enumerate(dataloader):
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            score = self.model(x)
            y_preds.extend(score.squeeze().tolist())
            y_trues.extend(y.squeeze().tolist())

        return torch.tensor(y_preds), torch.tensor(y_trues)

    @torch.no_grad()
    def report(self, y_trues: torch.Tensor, y_preds: torch.Tensor):
        report_path = os.path.join(self.config.output_dir, "report.txt")
        with open(report_path, "w") as f:
            thresholds = [(1.282, "80%"), (1.960, "95%"), (2.576, "99%"), (4.417, "99.999%")]
            for (thres, percent) in thresholds:
                result = metrics.classification_report(y_trues, y_preds > thres, digits=4)
                output = f"threshold: {thres} ({percent})\n{result}\n"
                f.write(output)
                print(output)

        fig = plt.figure()
        ax_pr = fig.add_subplot(3, 1, 1)
        ax_hist1 = fig.add_subplot(3, 1, 2)
        ax_hist2 = fig.add_subplot(3, 1, 3)

        p, r, t = metrics.precision_recall_curve(y_trues, y_preds)
        self._plot_prec_recall_vs_tresh(ax_pr, p, r, t)

        score_cls0 = y_preds[y_trues == 0]
        score_cls1 = y_preds[y_trues == 1]
        self._plot_histgram(ax_hist1, score_cls0, score_cls1)
        self._plot_histgram(ax_hist2, score_cls0, score_cls1, is_zoom=True)

        output_path = os.path.join(self.config.output_dir, "class_histgram.jpg")
        fig.savefig(output_path)

    def _plot_prec_recall_vs_tresh(self, ax, precisions, recalls, thresholds):
        ax.plot(thresholds, precisions[:-1], "b--", label="precision")
        ax.plot(thresholds, recalls[:-1], "g--", label="recall")
        ax.set_xlabel("Threshold")
        ax.legend(loc="upper right")
        ax.set_ylim([0, 1])

    def _plot_histgram(self, ax, score_cls0, score_cls1, is_zoom=False):
        bins = np.arange(-1.5, 8.0, 0.25)
        # bins = np.arange(-0.3, 0.3, 0.002)
        ax.hist(
            [score_cls0, score_cls1],
            bins=bins,
            color=["blue", "red"],
            label=["class 0", "class 1"],
            stacked=True,
        )
        locs = ax.get_yticks()
        ymax = locs[-1]
        if is_zoom:
            ymax /= 10
        ax.set_ylim([0, ymax])
        ax.set_yticks(np.arange(0, ymax, ymax / 5))
        ax.vlines(x=[1.282, 1.960, 2.576, 4.417], ymin=0, ymax=ymax)
        ax.grid(True)
