import torch
import torch.nn as nn

from common_util import get_logger
from config import Config
from prettytable import PrettyTable

logger = get_logger(__name__)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    logger.info(table)
    logger.info(f"Total Trainable Params: {total_params}")
    return total_params


def load_model(
    config: Config, model_file: str, device: torch.device = torch.device("cuda")
) -> nn.Module:
    # Loading Model
    from vlm_img_lang_model import ImgLanguageModel

    target_model = ImgLanguageModel(config=config)
    checkpoint = torch.load(model_file, weights_only=False)
    print(checkpoint.keys())
    target_model.load_state_dict(checkpoint["model_state_dict"])
    target_model = target_model.to(device)
    return target_model
