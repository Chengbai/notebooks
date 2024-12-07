import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader


@dataclass
class TrainSettings:
    batch_size: int = 100
    epoches: int = 8
    eval_interval_steps: int = 500
    gradient_accum_steps: int = 10
    gradient_accum_lr_scaler: float = 1.0  # 20.0

    eval_steps: int = 10
    lr: float = 5e-4 * gradient_accum_lr_scaler
    max_l2_grad_norm: int = 2

    train_dataloader: DataLoader = None
    eval_dataloader: DataLoader = None
    test_dataloader: DataLoader = None

    optimizer = None
    scheduler = None

    train_accuracy_momentum: float = 0.9
    eval_accuracy_momentum: float = 0.9

    device = torch.device("cpu")

    def validate(self):
        assert self.eval_interval_steps % self.gradient_accum_steps == 0

    def to_json(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "epoches": self.epoches,
            "eval_interval_steps": self.eval_interval_steps,
            "gradient_accum_steps": self.gradient_accum_steps,
            "gradient_accum_lr_scaler": self.gradient_accum_lr_scaler,
            "eval_steps": self.eval_steps,
            "lr": self.lr,
            "max_l2_grad_norm": self.max_l2_grad_norm,
        }
