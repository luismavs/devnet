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
    def __init__(self, n_input):
        super(Model, self).__init__()
        self.fc = nn.Linear(n_input, 20)
        self.output = nn.Linear(20, 1)

        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.output.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, x):
        x = F.relu(self.fc(x))
        return self.output(x)


class Phase(Enum):
    TRAIN = auto()
    EVAL = auto()
    PREDICT = auto()


class TableDataset(Dataset):
    def __init__(self, config: TrainerConfig, phase: Phase, logger: Logger) -> None:
        self.config = config
        self.phase = phase
        self.logger = logger

        path = os.path.join(config.dataroot, f"{phase.name.lower()}.csv")
        df = pd.read_csv(path)

        if "class" not in df.columns:
            assert False, f"columns of 'class' is necesasry in {path}"

        self.xs = torch.tensor(
            df.drop(columns=["class"]).values, dtype=torch.float32, requires_grad=True
        )
        self.ys = torch.tensor(df.loc[:, "class"].values, dtype=torch.int8)
        self.logger.info("==============================")
        self.logger.info(f"TableDataset {phase.name}")
        self.logger.info(f"data shape: {self.xs.shape}")
        self.logger.info(f"n_inliner: {torch.sum(self.ys == 0)}")
        self.logger.info(f"n_outliner: {torch.sum(self.ys == 1)}")
        self.logger.info("==============================")

        self._column_names = df.columns

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]

    def __len__(self):
        return len(self.ys)

    @property
    def n_columns(self):
        return self.xs.shape[1]

    @property
    def column_names(self):
        return self._column_names


class BalancedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self, dataset: TableDataset, n_batch: int, batch_size: int, seed: int
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
        indices = torch.randint(
            high=len(data), size=(self.n_samples_per_class,), generator=self.gen
        )
        return data[indices].tolist()


class Trainer:
    def __init__(self, config: TrainerConfig, logger: Optional[Logger] = None) -> None:
        self.config = config
        self.logger = getLogger(__name__) if logger is None else logger

        dataset_train = TableDataset(self.config, Phase.TRAIN, self.logger)
        dataset_eval = TableDataset(self.config, Phase.EVAL, self.logger)
        assert (
            dataset_train.n_columns == dataset_eval.n_columns
        ), "n_columns should be same in train and eval"

        self.dataloader_train = self._create_dataloader(dataset_train)
        self.dataloader_eval = self._create_dataloader(dataset_eval)

        torch.manual_seed(config.random_seed)
        self.model = Model(dataset_train.n_columns)
        self.optimizer = optim.RMSprop(
            self.model.parameters(), lr=0.001, alpha=0.9, eps=1e-7, weight_decay=0.01
        )

    def _create_dataloader(self, dataset: TableDataset) -> DataLoader:
        if dataset.phase == Phase.EVAL:
            return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        sampler = BalancedBatchSampler(
            dataset,
            self.config.n_batch,
            self.config.batch_size,
            self.config.random_seed,
        )
        return torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

    def forward(self, x, y):
        y_pred = self.model(x).squeeze()

        ref = torch.normal(mean=0, std=1, size=(5000,))
        score = (y_pred - torch.mean(ref)) / torch.std(ref)

        inlier = (1 - y) * torch.abs(score)
        outlier = y * torch.maximum(torch.zeros_like(score), 5 - score)

        return torch.mean(inlier + outlier)

    def _train(self) -> float:
        self.model.train()

        losses = []

        for i, (x, y) in enumerate(self.dataloader_train):
            x = x.to(self.config.device)
            y = y.to(self.config.device)

            loss = self.forward(x, y)
            loss.backward()
            losses.append(loss.item())

            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return sum(losses) / len(losses)

    def train(self) -> None:
        for epoch in range(self.config.epochs):
            start_time = time.time()
            loss = self._train()

            if epoch % self.config.log_interval == 0:
                elapsed_time = time.time() - start_time
                self.logger.info(
                    "[train] epoch: {}, loss: {:.2f}, time: {:.2f}".format(
                        epoch, loss, elapsed_time
                    )
                )

            if epoch % self.config.eval_interval == 0:
                self.eval(epoch)

        self.eval(-1, is_report=True)

    @torch.no_grad()
    def eval(self, epoch: int, is_report=False) -> None:
        self.model.eval()

        y_preds = []
        y_trues = []

        for i, (x, y) in enumerate(self.dataloader_eval):
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            score = self.model(x)
            y_preds.extend(score.squeeze().tolist())
            y_trues.extend(y.squeeze().tolist())

        y_preds = torch.tensor(y_preds)
        y_trues = torch.tensor(y_trues)

        roc_auc = metrics.roc_auc_score(y_trues, y_preds)
        ap = metrics.average_precision_score(y_trues, y_preds)
        self.logger.info(
            f"[eval] epoch: {epoch}, AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap)
        )

        if is_report:
            self.report(y_trues, y_preds)

    @torch.no_grad()
    def report(self, y_trues: torch.Tensor, y_preds: torch.Tensor):
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

        plt.show()

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
