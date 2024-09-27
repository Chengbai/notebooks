import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from caption_util import plot_caption_pred
from common_util import get_logger
from config import Config
from datetime import datetime
from dataclasses import dataclass, asdict
from fliker_comment_tokenizer import FlikerCommentTokenizer
from img_comment_dataset import ImgCommentDataset
from img_embedding import ImageEmbedding
from img_transformer import ImgTransformer
from img_util import show_img_tensor_CHW
from model_util import count_parameters
from pathlib import Path
from text_casual_mask_transformer import TextMaskedTransformer
from text_token_embedding import TextTokenEmbedding
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import List, Tuple
from vlm_model import ImgLanguageModel
from train_settings import TrainSetting

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
import torchvision.transforms.functional as VF

logger = get_logger(__name__)


def init_dataloaders(config: Config, train_setting: TrainSetting):
    split_portions: Tuple[float, float] = (0.72, 0.18, 0.1)
    train_dataset = ImgCommentDataset(
        config, split="train", split_portions=split_portions
    )
    eval_dataset = ImgCommentDataset(
        config, split="eval", split_portions=split_portions
    )
    test_dataset = ImgCommentDataset(
        config, split="test", split_portions=split_portions
    )
    logger.info(f"train_dataset:  {len(train_dataset)}")
    logger.info(f"eval_dataset:  {len(eval_dataset)}")
    logger.info(f"test_dataset:  {len(test_dataset)}")

    # Data Loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_setting.batch_size,
        shuffle=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=train_setting.batch_size, shuffle=True, num_workers=2
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=train_setting.batch_size, shuffle=True, num_workers=1
    )
    logger.info(f"train_dataloader:  {len(train_dataloader)}")
    logger.info(f"eval_data_loader:  {len(eval_dataloader)}")
    logger.info(f"test_data_loader:  {len(test_dataloader)}")

    train_setting.train_dataloader = train_dataloader
    train_setting.eval_dataloader = eval_dataloader
    train_setting.test_dataloader = test_dataloader

    return train_dataloader, eval_dataloader, test_dataloader


# model
def create_model(config: Config, train_setting: TrainSetting):
    model = ImgLanguageModel(config=config)
    (
        batch_aug_img_tensor1,
        batch_aug_img_tensor2,
        batch_img_id_tensor,
        batch_comment_encoding,
        batch_text_mask,
    ) = next(iter(train_setting.train_dataloader))
    logger.info(f"batch_aug_img_tensor1: {batch_aug_img_tensor1.size()}")
    logger.info(f"batch_aug_img_tensor2: {batch_aug_img_tensor2.size()}")
    logger.info(f"batch_img_id_tensor: {batch_img_id_tensor.size()}")
    logger.info(f"batch_comment_encoding: {batch_comment_encoding.size()}")
    logger.info(f"batch_text_mask: {batch_text_mask.size()}")
    (
        img_img_loss,
        img_text_loss,
        text_img_loss,
        img_img_contrastive_prob,
        img_text_contrastive_prob,
        text_img_contrastive_prob,
        lm_loss,
        lm_logit,
    ) = model(
        batch_aug_img_tensor1=batch_aug_img_tensor1,
        batch_aug_img_tensor2=batch_aug_img_tensor2,
        batch_text_tensor=batch_comment_encoding,
        batch_text_mask_tensor=batch_text_mask,
        batch_img_id_tensor=batch_img_id_tensor,
    )
    logger.info(f"img_img_loss: {img_img_loss}")
    logger.info(f"img_text_loss: {img_text_loss}")
    logger.info(f"text_img_loss: {text_img_loss}")
    logger.info(f"lm_loss: {lm_loss}")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f"pytorch_total_params: {pytorch_total_params/10**6} m")
    logger.info(
        f"pytorch_total_trainable_params: {pytorch_total_trainable_params/10**6} m"
    )
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
                batch_aug_img_tensor1,
                batch_aug_img_tensor2,
                batch_img_id_tensor,
                batch_text_tensor,
                batch_text_mask_tensor,
            ) = data
            batch_aug_img_tensor1 = batch_aug_img_tensor1.to(train_setting.device)
            batch_aug_img_tensor2 = batch_aug_img_tensor2.to(train_setting.device)
            batch_img_id_tensor = batch_img_id_tensor.to(train_setting.device)
            batch_text_tensor = batch_text_tensor.to(train_setting.device)
            batch_text_mask_tensor = batch_text_mask_tensor.to(train_setting.device)

            (
                img_img_loss,
                img_text_loss,
                text_img_loss,
                img_img_contrastive_prob,
                img_text_contrastive_prob,
                text_img_contrastive_prob,
                lm_loss,
                lm_logit,
            ) = model(
                batch_aug_img_tensor1=batch_aug_img_tensor1,
                batch_aug_img_tensor2=batch_aug_img_tensor2,
                batch_text_tensor=batch_text_tensor,
                batch_text_mask_tensor=batch_text_mask_tensor,
                batch_img_id_tensor=batch_img_id_tensor,
            )

            img_pred = torch.argmax(img_text_contrastive_prob, dim=1).cpu()
            label_mask = torch.arange(img_pred.size()[0]).cpu()
            img_accuracy = img_pred == label_mask
            img_accuracies.extend(img_accuracy)

            text_pred = torch.argmax(text_img_contrastive_prob, dim=1).cpu()
            text_accuracy = text_pred == label_mask
            text_accuracies.extend(text_accuracy)

            # Loss
            writer.add_scalar("eval/Img-Img Loss", img_img_loss, global_step)
            writer.add_scalar("eval/Img-Text Loss", img_text_loss, global_step)
            writer.add_scalar("eval/Text-Img Loss", text_img_loss, global_step)
            writer.add_scalar("eval/LM Loss", lm_loss, global_step)
            eval_loss = img_img_loss + img_text_loss + text_img_loss + lm_loss
            eval_losses.append(eval_loss)
            writer.add_scalar(
                "eval/Loss",
                eval_loss,
                global_step,
            )

            # Weighted Loss
            weighted_img_img_loss = config.img_img_loss_weight * img_img_loss
            writer.add_scalar(
                "weighted eval/Img-Img Loss Weight",
                config.img_img_loss_weight,
                global_step,
            )
            writer.add_scalar(
                "weighted eval/Img-Img Loss",
                weighted_img_img_loss,
                global_step,
            )
            weighted_img_text_loss = config.img_text_loss_weight * img_text_loss
            writer.add_scalar(
                "weighted eval/Img-Text Loss Weight",
                config.img_text_loss_weight,
                global_step,
            )
            writer.add_scalar(
                "weighted eval/Img-Text Loss",
                weighted_img_text_loss,
                global_step,
            )
            weighted_text_img_loss = config.text_img_loss_weight * text_img_loss
            writer.add_scalar(
                "weighted eval/Text-Img Loss Weight",
                config.text_img_loss_weight,
                global_step,
            )
            writer.add_scalar(
                "weighted eval/Text-Img Loss",
                weighted_text_img_loss,
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
            weighted_eval_loss = (
                weighted_img_img_loss
                + weighted_img_text_loss
                + weighted_text_img_loss
                + weighted_lm_loss
            )
            writer.add_scalar(
                "weighted eval/Loss",
                weighted_eval_loss,
                global_step,
            )
            weighted_eval_losses.append(weighted_eval_loss)

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

        # Caption
        img_file_paths = [
            "/Users/chengbai/ml/dataset/coco/train2017/000000233075.jpg",
            "/Users/chengbai/ml/dataset/coco/train2017/000000233111.jpg",
            "/Users/chengbai/ml/dataset/coco/train2017/000000348935.jpg",
            "/Users/chengbai/ml/dataset/flickr30k_images/flickr30k_images/3273585735.jpg",
            "/Users/chengbai/ml/dataset/flickr30k_images/flickr30k_images/428501119.jpg",
            "/Users/chengbai/ml/dataset/flickr30k_images/flickr30k_images/4282367460.jpg",
        ]

        fig = plot_caption_pred(
            img_langualge_model=model,
            img_file_paths=img_file_paths,
            device=train_setting.device,
        )

        writer.add_figure("Caption prediction", fig, global_step=global_step)

    model.train()
    writer.flush()
    return avg_eval_loss, eval_loss_std


def log_gradients_in_model(
    model: nn.Module,
    parameter_names: List[str],
    writer: SummaryWriter,
    global_step: int,
):
    if not parameter_names:
        return
    for pname, value in model.named_parameters():
        if pname in parameter_names and value.grad is not None:
            writer.add_histogram(pname + "/grad", value.grad.cpu(), global_step)


def train(
    model: ImgLanguageModel,
    config: Config,
    train_setting: TrainSetting,
    writer: SummaryWriter,
    start_epoch: int = 0,
    start_global_step: int = 0,
    debug: bool = False,
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
        running_img_accuracy = torch.tensor(
            0, dtype=torch.float32, device=torch.device("cpu")
        )
        running_text_accuracy = torch.tensor(
            0, dtype=torch.float32, device=torch.device("cpu")
        )

        logger.info(f"start_epoch: {start_epoch}")
        logger.info(f"start_global_step: {start_global_step}")
        # with torch.mps.profiler.profile(mode="interval", wait_until_completed=False):
        for epoch in range(train_setting.epoches):
            if epoch < start_epoch:
                continue
            train_dataloader_len = len(train_setting.train_dataloader)
            for train_step, data in enumerate(train_setting.train_dataloader):
                global_step = (
                    epoch * len(train_setting.train_dataloader)
                    + train_step
                    + start_global_step
                )

                writer.add_scalar("epch", epoch, global_step)

                # Profile
                if global_step < 1 + 1 + 3:
                    prof.step()

                (
                    batch_aug_img_tensor1,
                    batch_aug_img_tensor2,
                    batch_img_id_tensor,
                    batch_text_tensor,
                    batch_text_mask_tensor,
                ) = data
                batch_aug_img_tensor1 = batch_aug_img_tensor1.to(train_setting.device)
                batch_aug_img_tensor2 = batch_aug_img_tensor2.to(train_setting.device)
                batch_img_id_tensor = batch_img_id_tensor.to(train_setting.device)
                batch_text_tensor = batch_text_tensor.to(train_setting.device)
                batch_text_mask_tensor = batch_text_mask_tensor.to(train_setting.device)

                # Viz Model
                # if global_step == 0:
                #     writer.add_graph(model, (batch_img_tensor, batch_target_tensor))

                (
                    img_img_loss,
                    img_text_loss,
                    text_img_loss,
                    img_img_contrastive_prob,
                    img_text_contrastive_prob,
                    text_img_contrastive_prob,
                    lm_loss,
                    lm_logit,
                ) = model(
                    batch_aug_img_tensor1=batch_aug_img_tensor1,
                    batch_aug_img_tensor2=batch_aug_img_tensor2,
                    batch_text_tensor=batch_text_tensor,
                    batch_text_mask_tensor=batch_text_mask_tensor,
                    batch_img_id_tensor=batch_img_id_tensor,
                )

                # Loss
                writer.add_scalar("train/Img-Img Loss", img_img_loss, global_step)
                writer.add_scalar("train/Img-Text Loss", img_text_loss, global_step)
                writer.add_scalar("train/Text-Img Loss", text_img_loss, global_step)
                writer.add_scalar("train/LM Loss", lm_loss, global_step)
                writer.add_scalar(
                    "train/Loss",
                    img_img_loss + img_text_loss + text_img_loss + lm_loss,
                    global_step,
                )

                # Weighted Loss
                weighted_img_img_loss = config.img_img_loss_weight * img_img_loss
                writer.add_scalar(
                    "weighted train/Img-Img Loss Weight",
                    config.img_img_loss_weight,
                    global_step,
                )
                writer.add_scalar(
                    "weighted train/Img-Img Loss",
                    weighted_img_img_loss,
                    global_step,
                )
                weighted_img_text_loss = config.img_text_loss_weight * img_text_loss
                writer.add_scalar(
                    "weighted train/Img-Text Loss Weight",
                    config.img_text_loss_weight,
                    global_step,
                )
                writer.add_scalar(
                    "weighted train/Img-Text Loss",
                    weighted_img_text_loss,
                    global_step,
                )
                weighted_text_img_loss = config.text_img_loss_weight * text_img_loss
                writer.add_scalar(
                    "weighted train/Text-Img Loss Weight",
                    config.text_img_loss_weight,
                    global_step,
                )
                writer.add_scalar(
                    "weighted train/Text-Img Loss",
                    weighted_text_img_loss,
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

                train_loss = (
                    weighted_img_img_loss
                    + weighted_img_text_loss
                    + weighted_text_img_loss
                    + weighted_lm_loss
                )
                writer.add_scalar(
                    "weighted train/Loss",
                    train_loss,
                    global_step,
                )

                writer.add_scalar(
                    "Learning Rate",
                    train_setting.scheduler.get_last_lr()[-1],
                    global_step,
                )
                loss = (train_loss) / train_setting.gradient_agg_steps

                loss.backward()

                if debug:
                    log_gradients_in_model(
                        model=model,
                        parameter_names=["img_embedding.conv.weight"],
                        writer=writer,
                        global_step=global_step,
                    )

                if ((global_step + 1) % train_setting.gradient_agg_steps == 0) or (
                    global_step + 1 == train_dataloader_len
                ):
                    # nn.utils.clip_grad_norm_(
                    #     model.parameters(), max_norm=train_setting.max_l2_grad_norm
                    # )
                    train_setting.optimizer.step()
                    train_setting.optimizer.zero_grad()

                    for _ in range(train_setting.gradient_agg_steps):
                        train_setting.scheduler.step()

                # Performance
                img_pred = torch.argmax(img_text_contrastive_prob, dim=1).cpu()
                label_mask = torch.arange(img_pred.size()[0]).cpu()
                img_accuracy = img_pred == label_mask
                img_accuracy = img_accuracy.float().mean()
                running_img_accuracy = (
                    train_setting.train_accuracy_momentum * running_img_accuracy
                    + (1 - train_setting.train_accuracy_momentum) * img_accuracy
                )

                text_pred = torch.argmax(text_img_contrastive_prob, dim=1).cpu()
                text_accuracy = text_pred == label_mask
                text_accuracy = text_accuracy.float().mean()
                running_text_accuracy = (
                    train_setting.train_accuracy_momentum * running_text_accuracy
                    + (1 - train_setting.train_accuracy_momentum) * text_accuracy
                )
                writer.add_scalar(
                    "perf/Train Img Accuracy",
                    running_img_accuracy,
                    global_step,
                )
                writer.add_scalar(
                    "perf/Train Text Accuracy",
                    running_text_accuracy,
                    global_step,
                )

                writer.add_scalar(
                    "cache/Rolling Cache Enabled",
                    config.rolling_cache_enabled,
                    global_step,
                )

                writer.add_scalar(
                    "cache/Rolling Cache Size",
                    config.rolling_cache_size,
                    global_step,
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

                if (
                    train_step > 0
                    and train_step % train_setting.eval_interval_steps == 0
                ):
                    assert (
                        train_setting.eval_interval_steps
                        % train_setting.gradient_agg_steps
                        == 0
                    ), ""
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
                        model_path = (
                            f"vlm_caption_model_{epoch}_{timestamp}_{global_step}.pt"
                        )
                        torch.save(
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "total_steps": len(train_setting.train_dataloader)
                                * train_setting.epoches,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": train_setting.optimizer.state_dict(),
                                "loss": best_vloss,
                                "config": asdict(config),
                                "train_settings": asdict(train_setting),
                            },
                            model_path,
                        )


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


def init_train_device(train_setting: TrainSetting):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        torch.device("cpu")

    train_setting.device = device
    return device


def load_checkpoint(
    model_path: str,
    override_config: Config = None,
    override_train_setting: TrainSetting = None,
) -> Tuple[nn.Module, torch.optim.Optimizer, Config, TrainSetting, int, int]:
    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False)
    logger.info(checkpoint.keys())

    # Load Config
    if override_config is not None:
        config = override_config
    else:
        config = Config(**checkpoint["config"])

    # Load TrainSetting and
    if override_train_setting is not None:
        train_setting = override_train_setting
    else:
        train_setting = TrainSetting(**checkpoint["train_settings"])
    init_train_device(train_setting=train_setting)
    logger.info(f"train_setting.device: {train_setting.device}")

    # Load Model
    model_trained = ImgLanguageModel(config=config)
    model_trained.load_state_dict(checkpoint["model_state_dict"])
    model_trained = model_trained.to(train_setting.device)

    # Load optimizer
    optimizer = create_optimizer(
        config=config, train_setting=train_setting, model=model_trained
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    assert optimizer is not None

    # Load training progress
    epoch = checkpoint["epoch"]
    global_step = checkpoint["global_step"]

    init_dataloaders(
        config=config,
        train_setting=train_setting,
    )

    return model_trained, optimizer, config, train_setting, epoch, global_step


def train_model(checkpoint: str = None, debug: bool = False):
    config = Config()

    if not checkpoint:
        train_setting = TrainSetting()

        init_train_device(train_setting=train_setting)

        init_dataloaders(
            config=config,
            train_setting=train_setting,
        )

        model = create_model(config=config, train_setting=train_setting)
        model = model.to(train_setting.device)
        optimizer = create_optimizer(
            config=config, train_setting=train_setting, model=model
        )
        assert optimizer is not None

        start_epoch = 0
        start_global_step = 0
    else:
        model, optimizer, config, train_setting, start_epoch, start_global_step = (
            load_checkpoint(model_path=checkpoint, override_config=config)
        )
        if train_setting.device != torch.device("mps"):
            model = torch.compile(model)

    scheduler = create_scheduler(config=config, train_setting=train_setting)
    assert scheduler is not None

    with SummaryWriter(flush_secs=1) as writer:
        train(
            model=model,
            config=config,
            train_setting=train_setting,
            writer=writer,
            start_epoch=start_epoch,
            start_global_step=start_global_step,
            debug=debug,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"vlm_caption_model_{timestamp}_final.pt"
        torch.save(
            {
                "epoch": train_setting.epoches,  # final epoch
                "total_steps": len(train_setting.train_dataloader)
                * train_setting.epoches,
                "global_step": len(train_setting.train_dataloader)
                * train_setting.epoches,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
                "train_settings": asdict(train_setting),
            },
            model_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLM model training.")
    parser.add_argument(
        "--checkpoint", help="Start model training from given checkpoint", default=""
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.checkpoint:
        logger.info(f"Start model training from checkpoint: {args.checkpoint}.")

    if args.debug:
        logger.info("Running in debug mode.")

    train_model(checkpoint=args.checkpoint, debug=args.debug)
