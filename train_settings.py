import torch
from dataclasses import asdict, dataclass, field
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainSettings:
    batch_size = 20
    epoches = 5
    eval_interval_steps = 500
    gradient_accum_steps = 100
    eval_steps = 10
    lr = 5e-4
    max_l2_grad_norm = 2

    train_dataloader: DataLoader = None
    eval_dataloader: DataLoader = None
    test_dataloader: DataLoader = None

    optimizer = None
    scheduler = None

    train_accuracy_momentum = 0.9
    eval_accuracy_momentum = 0.9

    device = torch.device("cpu")

    def validate(self):
        assert self.eval_interval_steps % self.gradient_accum_steps == 0

    def to_json(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "epoches": self.epoches,
            "eval_interval_steps": self.eval_interval_steps,
            "gradient_accum_steps": self.gradient_accum_steps,
            "eval_steps": self.eval_steps,
            "lr": self.lr,
            "max_l2_grad_norm": self.max_l2_grad_norm,
        }
