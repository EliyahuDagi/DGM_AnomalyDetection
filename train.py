from __future__ import annotations
from typing import Dict, Tuple
from tqdm import tqdm
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CyclicLR
from model_interface import GenerativeModel
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
import os
import torch


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.n_count = 0
        self.sum = 0

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def train(model_name: str, model: GenerativeModel,
          dataset_name: str, datasets: Tuple[Dataset, Dataset],
          cfg: Dict, output_dir):
    train_dataset, valid_dataset = datasets
    train_loader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.valid_batch_size, shuffle=False,
                              num_workers=cfg.num_workers)

    optimizer = Adam(model.trainable_params(), lr=1e-4)

    scheduler = CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2, cycle_momentum=False, step_size_up=len(train_loader))
    train_loss_avg = Averager()
    val_loss_avg = Averager()
    out_dir = os.path.join(output_dir, model_name, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    lowest_loss = float('inf')
    writer = SummaryWriter(os.path.join(out_dir, 'tensorboard'), 'tensorboard')
    samples_dir = os.path.join(out_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    model = model.to(model.device)
    for epoch_index in tqdm(range(cfg.num_epochs), desc='epoch', leave=False, colour='green', ncols=80):
        model.train()
        model = model.to(model.device)
        train_loss_avg.reset()
        for iter_idx, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train on {dataset_name}', leave=False, colour='red', ncols=80):
            optimizer.zero_grad()
            model_out = model(x.to(model.device))
            loss = model.criterion(**model_out)
            train_loss_avg.add(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
        writer.add_scalar('Loss/train', train_loss_avg.val(), epoch_index)
        model.eval()
        val_loss_avg.reset()
        with torch.no_grad():
            for x, _ in tqdm(valid_loader, desc=f'validation on {dataset_name}', leave=False, colour='red', ncols=80):
                model_out = model(x.to(model.device))
                loss = model.criterion(**model_out)
                val_loss_avg.add(loss)
        val_loss = val_loss_avg.val()
        print(f'epoch {epoch_index} train loss = {train_loss_avg.val().item()}, val loss = {val_loss}')
        writer.add_scalar('Loss/val', val_loss, epoch_index)
        if val_loss < lowest_loss:
            torch.save(model.state_dict(), os.path.join(out_dir, 'model.pth'))
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pth'))