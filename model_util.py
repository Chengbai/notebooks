from common_util import get_logger
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
