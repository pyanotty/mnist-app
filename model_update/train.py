# coding: UTF-8

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow

DATA_ROOT = "."
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

from logging import getLogger
logger = getLogger(__name__)

class MNISTNet(nn.Module):
  def __init__(self, in_features, dim_hidden1, dim_hidden2, out_features):
      super().__init__()

      self.hiddens = nn.Sequential(
          nn.Linear(in_features, dim_hidden1),
          nn.ReLU(),
          nn.Linear(dim_hidden1, dim_hidden2),
          nn.ReLU(),
      )

      self.cls = nn.Linear(dim_hidden2, out_features)

  def forward(self, x):
      out = self.hiddens(x)
      out = self.cls(out)
      return out


def train_one_epoch(dataloader, model, optimizer, criterion):
  losses = []

  model.to(DEVICE)

  # dropoutなど、学習時と推論時とモデルの挙動が異なる場合がある
  # ので、学習を行うことを明に宣言すると良い
  model.train()

  prog_bar = tqdm(dataloader)
  for X, y in prog_bar:
    # 初めに勾配を0で初期化する
    # これをしないと勾配が蓄積していってヤバいことが起きる
    optimizer.zero_grad()

    # 入力とモデルは同じDEVICE上にある必要がある
    X = X.to(DEVICE)
    y = y.reshape(-1).to(DEVICE)

    # forward
    out = model(X)
    loss = criterion(out, y)
    # backward
    loss.backward()
    # step
    optimizer.step()

    # detach()によって勾配情報を含まない形でテンソルを取り出せる
    losses.append(
      loss.detach().cpu()
    )
    # message = f"train loss: {np.mean(losses):.5f}"
    message = "hoge"
    prog_bar.set_description(message)

  return np.mean(losses)


def valid_one_epoch(dataloader, model, criterion):
  losses = []

  model.eval()
  model.to(DEVICE)

  prog_bar = tqdm(dataloader)
  # 勾配計算を行わないことを明示的に宣言することで、計算が速くなる
  with torch.no_grad():

    for X, y in prog_bar:

      X = X.to(DEVICE)
      y = y.reshape(-1).to(DEVICE)

      out = model(X)
      loss = criterion(out, y)

      losses.append(loss.detach().cpu())
      # message = f"valid loss: {np.mean(losses):.5f}"
      message = "hoge"
      prog_bar.set_description(message)

  return np.mean(losses)


@hydra.main(config_path="./", config_name="config")
def do_train(cfg: DictConfig) -> None:

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.log_param("epochs", cfg.epochs)


    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform,
    )

    valid_dataset = datasets.MNIST(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    in_features = 28 * 28
    out_features = 10

    model = MNISTNet(
        in_features=in_features,
        dim_hidden1=cfg.dim_hidden1,
        dim_hidden2=cfg.dim_hidden2,
        out_features=out_features,
    )
    optimizer = Adam(params=model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_valid_loss = np.inf

    for epoch in range(cfg.epochs):
        loss_train = train_one_epoch(
            dataloader=train_loader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
        )

        loss_valid = valid_one_epoch(
            dataloader=valid_loader, model=model, criterion=criterion
        )

        mlflow.log_metric("loss_valid", loss_valid)
        if loss_valid < best_valid_loss:
            torch.save(model.state_dict(), "/model_update/model.pth")
            
            logger.info(f"Validation loss is improved: {best_valid_loss} -> {loss_valid}")
            best_valid_loss = loss_valid
        mlflow.log_artifact("/model_update/model.pth")
        mlflow.log_artifact("/model_update/train.py")
        mlflow.log_param("hoge", "fuga")
        # mlflow.pytorch.log_model(model, "model", registered_model_name="mnist-model")

    return 0

if __name__ == "__main__":
  do_train()
