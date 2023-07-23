import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Union


class C2M(nn.Module):
    def __init__(self, num_stats: int, hidden_size: Union[List[int], int], out_features: int, num_hiddens: int = 0):
        super().__init__()
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        self.fc_layers = []
        self.fc_layers.extend([nn.Linear(num_stats, hidden_size[0]), nn.ReLU()])  # , nn.Dropout(0.2)])
        if num_hiddens > 0:
            for i in range(num_hiddens):
                self.fc_layers.extend([nn.Linear(hidden_size[i], hidden_size[i + 1]), nn.ReLU()])
        self.fc_layers = nn.Sequential(*self.fc_layers)
        self.out_layer = nn.Linear(hidden_size[-1], out_features)

    def forward(self, x):
        x = self.fc_layers(x)

        return self.out_layer(x)


class C2M_pl(pl.LightningModule):
    def __init__(
        self,
        num_stats: int,
        labels: List[List[str]],
        lr: float = 0.0001,
        out_features: int = 10,
        hidden_size: Union[int, List[int]] = 300,
        num_hiddens: int = 0,
    ):
        super().__init__()
        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        self.save_hyperparameters()
        self.model = C2M(
            num_stats=num_stats, out_features=out_features, hidden_size=hidden_size, num_hiddens=num_hiddens
        )
        self.lr = lr
        self.out_features = out_features
        self.labels = labels

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx):
        parameters, clip_labels = batch
        b = parameters.shape[0]
        parameters_pred = self(clip_labels)
        parameters_pred = parameters_pred.reshape(b, 1, self.out_features)
        loss = F.mse_loss(parameters, parameters_pred)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
