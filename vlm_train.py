import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from datetime import datetime
from img_embedding import ImageEmbedding
from img_transformer import ImgTransformer
from img_util import show_img_tensor_CHW
from fliker_comment_tokenizer import FlikerCommentTokenizer
from fliker_img_comment_dataset import ImgCommentDataset
from model_util import count_parameters
from pathlib import Path
from text_token_embedding import TextTokenEmbedding
from text_casual_mask_transformer import TextMaskedTransformer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from vlm_model import ImgLanguageModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
import torchvision.transforms.functional as VF


class TrainSetting:
    batch_size = 20
    epoches = 5
    eval_interval = 100
    eval_steps = 10
    lr = 5e-4
    max_l2_grad_norm = 2

    train_dataloader: DataLoader = None
    eval_dataloader: DataLoader = None
    test_dataloader: DataLoader = None

    optimizer = None
    scheduler = None

    device = torch.device("cpu")


def create_dataloaders(config: Config, train_setting: TrainSetting):
    train_dataset = ImgCommentDataset(config, split="train")
    eval_dataset = ImgCommentDataset(config, split="eval")
    test_dataset = ImgCommentDataset(config, split="test")
    print(f"train_dataset:  {len(train_dataset)}")
    print(f"eval_dataset:  {len(eval_dataset)}")
    print(f"test_dataset:  {len(test_dataset)}")

    # Data Loader
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_setting.batch_size, shuffle=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=train_setting.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=train_setting.batch_size, shuffle=True
    )
    print(f"train_dataloader:  {len(train_dataloader)}")
    print(f"eval_data_loader:  {len(eval_dataloader)}")
    print(f"test_data_loader:  {len(test_dataloader)}")

    train_setting.train_dataloader = train_dataloader
    train_setting.eval_dataloader = eval_dataloader
    train_setting.test_dataloader = test_dataloader

    return train_dataloader, eval_dataloader, test_dataloader


# model
def create_model(config: Config, train_setting: TrainSetting):
    model = ImgLanguageModel(config=config)
    batch_img_tensor, batch_img_id_tensor, batch_comment_encoding, batch_text_mask = (
        next(iter(train_setting.train_dataloader))
    )
    print(f"batch_img_tensor: {batch_img_tensor.size()}")
    print(f"batch_img_id_tensor: {batch_img_id_tensor.size()}")
    print(f"batch_comment_encoding: {batch_comment_encoding.size()}")
    print(f"batch_text_mask: {batch_text_mask.size()}")
    (
        img_loss,
        text_loss,
        img_contrastive_prob,
        text_contrastive_prob,
        lm_loss,
        lm_logit,
    ) = model(
        batch_img_tensor=batch_img_tensor,
        batch_text_tensor=batch_comment_encoding,
        batch_text_mask_tensor=batch_text_mask,
        batch_img_id_tensor=batch_img_id_tensor,
    )
    print(f"img_loss: {img_loss}")
    print(f"text_loss: {text_loss}")
    print(f"lm_loss: {lm_loss}")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"pytorch_total_params: {pytorch_total_params/10**6} m")
    print(f"pytorch_total_trainable_params: {pytorch_total_trainable_params/10**6} m")
    count_parameters(model)

    return model


def eval(
    model: ImgLanguageModel,
    config: Config,
    train_setting: TrainSetting,
    global_step: int,
    writer: SummaryWriter,
):
    model.eval()

    avg_eval_loss = None
    eval_loss_std = None
    img_accuracies = []
    text_accuracies = []
    with torch.no_grad():
        eval_losses = []
        weighted_eval_losses = []
        for i, data in enumerate(train_setting.eval_dataloader):
            if i > train_setting.eval_steps:
                # It takes significant time to do one full eval.
                break

            (
                batch_img_tensor,
                batch_img_id_tensor,
                batch_target_tensor,
                batch_target_mask,
            ) = data
            (
                batch_img_tensor,
                batch_img_id_tensor,
                batch_text_tensor,
                batch_text_mask_tensor,
            ) = data
            batch_img_tensor = batch_img_tensor.to(train_setting.device)
            batch_img_id_tensor = batch_img_id_tensor.to(train_setting.device)
            batch_text_tensor = batch_text_tensor.to(train_setting.device)
            batch_text_mask_tensor = batch_text_mask_tensor.to(train_setting.device)

            (
                img_loss,
                text_loss,
                img_contrastive_prob,
                text_contrastive_prob,
                lm_loss,
                lm_logit,
            ) = model(
                batch_img_tensor=batch_img_tensor,
                batch_text_tensor=batch_text_tensor,
                batch_text_mask_tensor=batch_text_mask_tensor,
                batch_img_id_tensor=batch_img_id_tensor,
            )

            img_pred = torch.argmax(img_contrastive_prob, dim=1).cpu()
            label_mask = torch.arange(img_pred.size()[0]).cpu()
            img_accuracy = img_pred == label_mask
            img_accuracies.extend(img_accuracy)

            text_pred = torch.argmax(text_contrastive_prob, dim=1).cpu()
            text_accuracy = text_pred == label_mask
            text_accuracies.extend(text_accuracy)

            # Loss
            writer.add_scalar("eval/Img Loss", img_loss, global_step)
            writer.add_scalar("eval/Text Loss", text_loss, global_step)
            writer.add_scalar("eval/LM Loss", lm_loss, global_step)
            eval_losses.append(img_loss + text_loss + lm_loss)

            # Weighted Loss
            weighted_img_loss = config.img_loss_weight * img_loss
            writer.add_scalar(
                "weighted eval/Img Loss Weight",
                config.img_loss_weight,
                global_step,
            )
            writer.add_scalar(
                "weighted eval/Img Loss",
                weighted_img_loss,
                global_step,
            )
            weighted_text_loss = config.text_loss_weight * text_loss
            writer.add_scalar(
                "weighted eval/Text Loss Weight",
                config.text_loss_weight,
                global_step,
            )
            writer.add_scalar(
                "weighted eval/Text Loss",
                weighted_text_loss,
                global_step,
            )
            weighted_lm_loss = config.lm_loss_weight * lm_loss
            writer.add_scalar(
                "weighted eval/LM Loss Weight",
                config.lm_loss_weight,
                global_step,
            )
            writer.add_scalar(
                "weighted eval/LM Loss",
                weighted_lm_loss,
                global_step,
            )
            writer.add_scalar(
                "weighted eval/Loss",
                weighted_img_loss + weighted_text_loss + weighted_lm_loss,
                global_step,
            )
            weighted_eval_losses.append(
                weighted_img_loss + weighted_text_loss + weighted_lm_loss
            )

        # Agg Loss
        eval_losses = torch.tensor(eval_losses)
        avg_eval_loss = eval_losses.mean()
        eval_loss_std = eval_losses.std()
        writer.add_scalar("eval/Loss", avg_eval_loss, global_step)
        writer.add_scalar("eval/eval-std", eval_loss_std, global_step)

        # Agg Weighted Loss
        weighted_eval_losses = torch.tensor(weighted_eval_losses)
        weighted_eval_loss_avg = weighted_eval_losses.mean()
        weighted_eval_loss_std = weighted_eval_losses.std()
        writer.add_scalar("weighted eval/Loss", weighted_eval_loss_avg, global_step)
        writer.add_scalar("weighted eval/eval-std", weighted_eval_loss_std, global_step)

        # Performance
        writer.add_scalar(
            "perf/Eval Img Accuracy",
            sum(img_accuracies) / len(img_accuracies),
            global_step,
        )
        writer.add_scalar(
            "perf/Eval Text Accuracy",
            sum(text_accuracies) / len(text_accuracies),
            global_step,
        )

    model.train()
    writer.flush()
    return avg_eval_loss, eval_loss_std


def train(
    model: ImgLanguageModel,
    config: Config,
    train_setting: TrainSetting,
    writer: SummaryWriter,
):
    best_vloss = torch.tensor(1_000_000)
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./runs"),
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # with torch.mps.profiler.profile(mode="interval", wait_until_completed=False):
        for epoch in range(train_setting.epoches):
            for train_step, data in enumerate(train_setting.train_dataloader):
                global_step = epoch * len(train_setting.train_dataloader) + train_step

                # Profile
                if global_step < 1 + 1 + 3:
                    prof.step()

                (
                    batch_img_tensor,
                    batch_img_id_tensor,
                    batch_text_tensor,
                    batch_text_mask_tensor,
                ) = data
                batch_img_tensor = batch_img_tensor.to(train_setting.device)
                batch_img_id_tensor = batch_img_id_tensor.to(train_setting.device)
                batch_text_tensor = batch_text_tensor.to(train_setting.device)
                batch_text_mask_tensor = batch_text_mask_tensor.to(train_setting.device)

                # Viz Model
                # if global_step == 0:
                #     writer.add_graph(model, (batch_img_tensor, batch_target_tensor))

                train_setting.optimizer.zero_grad()

                (
                    img_loss,
                    text_loss,
                    img_contrastive_prob,
                    text_contrastive_prob,
                    lm_loss,
                    lm_logit,
                ) = model(
                    batch_img_tensor=batch_img_tensor,
                    batch_text_tensor=batch_text_tensor,
                    batch_text_mask_tensor=batch_text_mask_tensor,
                    batch_img_id_tensor=batch_img_id_tensor,
                )

                # Loss
                writer.add_scalar("train/Img Loss", img_loss, global_step)
                writer.add_scalar("train/Text Loss", text_loss, global_step)
                writer.add_scalar("train/LM Loss", lm_loss, global_step)
                writer.add_scalar(
                    "train/Loss", img_loss + text_loss + lm_loss, global_step
                )

                # Weighted Loss
                weighted_img_loss = config.img_loss_weight * img_loss
                writer.add_scalar(
                    "weighted train/Img Loss Weight",
                    config.img_loss_weight,
                    global_step,
                )
                writer.add_scalar(
                    "weighted train/Img Loss",
                    weighted_img_loss,
                    global_step,
                )
                weighted_text_loss = config.text_loss_weight * text_loss
                writer.add_scalar(
                    "weighted train/Text Loss Weight",
                    config.text_loss_weight,
                    global_step,
                )
                writer.add_scalar(
                    "weighted train/Text Loss",
                    weighted_text_loss,
                    global_step,
                )
                weighted_lm_loss = config.lm_loss_weight * lm_loss
                writer.add_scalar(
                    "weighted train/LM Loss Weight",
                    config.lm_loss_weight,
                    global_step,
                )
                writer.add_scalar(
                    "weighted train/LM Loss",
                    weighted_lm_loss,
                    global_step,
                )
                writer.add_scalar(
                    "weighted train/Loss",
                    weighted_img_loss + weighted_text_loss + weighted_lm_loss,
                    global_step,
                )

                writer.add_scalar(
                    "Learning Rate",
                    train_setting.scheduler.get_last_lr()[-1],
                    global_step,
                )
                loss = weighted_img_loss + weighted_text_loss + weighted_lm_loss
                loss.backward()
                # ===============================================================================================================
                nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=train_setting.max_l2_grad_norm
                )
                # ===============================================================================================================
                # Error: command buffer exited with error status.
                # The Metal Performance Shaders operations encoded on it may not have completed.
                # Error:
                # (null)
                # Ignored (for causing prior/excessive GPU errors) (00000004:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)
                # <AGXG13XFamilyCommandBuffer: 0xa5e418420>
                # label = <none>
                # device = <AGXG13XDevice: 0x15430ee00>
                #     name = Apple M1 Max
                # commandQueue = <AGXG13XFamilyCommandQueue: 0x157a05800>
                #     label = <none>
                #     device = <AGXG13XDevice: 0x15430ee00>
                #         name = Apple M1 Max
                # retainedReferences = 1
                # ---------------------------------------------------------------------------------------------------------------
                train_setting.optimizer.step()
                train_setting.scheduler.step()

                if train_step > 0 and train_step % train_setting.eval_interval == 0:
                    avg_vloss, _ = eval(
                        model=model,
                        config=config,
                        train_setting=train_setting,
                        global_step=global_step,
                        writer=writer,
                    )

                    if avg_vloss is not None and avg_vloss < best_vloss:
                        best_vloss = avg_vloss
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_path = f"vlm_caption_model_{epoch}_{timestamp}"
                        torch.save(model.state_dict(), model_path)


def create_optimizer(config: Config, train_setting: TrainSetting, model: nn.Module):
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=train_setting.lr)
    # cqKMSkZKhJHGDHkcxSfU
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=epoches
    # )
    # ujlBwmQYUsBdDW
    train_setting.optimizer = optimizer
    return optimizer


def create_scheduler(config: Config, train_setting: TrainSetting):
    total_steps = train_setting.epoches * len(train_setting.train_dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        train_setting.optimizer,
        T_max=total_steps,
    )
    train_setting.scheduler = scheduler
    return scheduler


def get_train_device(train_setting: TrainSetting):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        torch.device("cpu")

    train_setting.device = device
    return device


def load_model(config: Config, train_setting: TrainSetting, model_path: str):
    model_trained = ImgLanguageModel(config=config)
    model_trained.load_state_dict(torch.load(model_path))
    model_trained = model_trained.to(train_setting.device)
    return model_trained


def train_model():
    config = Config()
    train_setting = TrainSetting()

    # fine turn
    # train_setting.lr = 1e-4

    device = get_train_device(train_setting=train_setting)

    create_dataloaders(
        config=config,
        train_setting=train_setting,
    )

    model = create_model(config=config, train_setting=train_setting)
    # model = load_model(
    #     config=config,
    #     train_setting=train_setting,
    #     model_path="/Users/chengbai/ml/cheng_git/notebooks/vlm_caption_model_20240905_023757_final",
    # )
    # if train_setting.device != torch.device("mps"):
    #     model = torch.compile(model)

    model = model.to(device)
    optimizer = create_optimizer(
        config=config, train_setting=train_setting, model=model
    )
    assert optimizer is not None

    scheduler = create_scheduler(config=config, train_setting=train_setting)
    assert scheduler is not None

    with SummaryWriter(flush_secs=1) as writer:
        train(model=model, config=config, train_setting=train_setting, writer=writer)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"vlm_caption_model_{timestamp}_final"
        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    train_model()
