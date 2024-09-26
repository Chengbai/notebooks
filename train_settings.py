import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader


@dataclass
class TrainSetting:
    batch_size = 20
    epoches = 5
    eval_interval_steps = 100
    eval_steps = 10
    lr = 5e-4
    max_l2_grad_norm = 2

    train_dataloader: DataLoader = None
    eval_dataloader: DataLoader = None
    test_dataloader: DataLoader = None

    optimizer = None
    scheduler = None

    gradient_agg_steps = 1

    train_accuracy_momentum = 0.9
    eval_accuracy_momentum = 0.9

    device = torch.device("cpu")
