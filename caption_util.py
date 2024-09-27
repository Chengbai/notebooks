import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torchvision.transforms.functional as VF

from common_util import get_logger
from config import Config
from fliker_comment_tokenizer import FlikerCommentTokenizer
from img_util import load_img_tensor, inverse_img_aug
from text_util import normalize_comment
from typing import List, Tuple
from vlm_model import ImgLanguageModel

logger = get_logger(__name__)


# Create Caption
def create_caption_from_aug_img_tensor(
    img_langualge_model: ImgLanguageModel,
    batch_aug_img_tensor: torch.tensor,
) -> str:
    assert img_langualge_model is not None
    assert batch_aug_img_tensor is not None
    assert not img_langualge_model.training, "Model should be set to eval mode."

    caption = None

    img_feature, img_contrastive_feature, img_feature_proj = (
        img_langualge_model.get_batch_img_feature(
            batch_aug_img_tensor=batch_aug_img_tensor
        )
    )

    bos_embedding = img_langualge_model.text_transformer.text_token_embedding(
        x=torch.tensor(
            img_langualge_model.bos_token, device=batch_aug_img_tensor.device
        ),
        skip_position_embedding=True,
    )

    comment_toeknizer = FlikerCommentTokenizer.get_tokenizer(
        config=img_langualge_model.config
    )
    caption_tokens = []  # comment_toeknizer.encode("<bos>")[:1]
    for index in range(img_langualge_model.config.max_text_len):
        normal_tokens, normal_mask = normalize_comment(
            config=img_langualge_model.config, comment_tokens=caption_tokens
        )
        batch_text_tensor = normal_tokens.to(device=batch_aug_img_tensor.device)[
            None, :
        ]  # 1 x MAX_TOKEN_LEN
        batch_text_mask_tensor = normal_mask.to(device=batch_aug_img_tensor.device)[
            None, :
        ]  # 1 x MAX_TOKEN_LEN

        text_feature = img_langualge_model.text_transformer(
            batch_text_tensor, need_embedding=True
        )
        text_feature = img_langualge_model.text_norm(text_feature)

        lm_logits, lm_loss = img_langualge_model.img_caption_model(
            img_feature=img_feature,
            text_feature=text_feature,
            text_mask=(batch_text_tensor != 0),
            batch_target_text_token=batch_text_tensor,
            target_text_mask_tensor=batch_text_mask_tensor,
            bos_embedding=bos_embedding,
        )
        # logger.info(f"lm_loss: {lm_loss}, normal_mask: {normal_mask}")
        cur_token = torch.argmax(lm_logits[0][index])
        caption_tokens.append(cur_token)
        if (
            cur_token == img_langualge_model.bos_token and len(caption_tokens) > 1
        ):  # skip the 1st <bos>
            break

    caption = comment_toeknizer.decode(caption_tokens)
    return caption


# Create Caption
def create_caption_from_img_file(
    img_langualge_model: ImgLanguageModel,
    img_file_path: str,
    device: torch.device,
) -> Tuple[PIL.Image.Image, str]:
    img_aug_tensor1, img_aug_tensor2 = load_img_tensor(
        config=img_langualge_model.config, img_file_path=img_file_path
    )
    img_aug_tensor1 = img_aug_tensor1.to(device)
    img_aug_tensor2 = img_aug_tensor2.to(device)
    img_org = inverse_img_aug(img_aug_tensor1)
    img_pil = VF.to_pil_image(img_org)

    caption = create_caption_from_aug_img_tensor(
        img_langualge_model=img_langualge_model,
        batch_aug_img_tensor=img_aug_tensor1[None, :],
    )
    return img_pil, caption


def plot_caption_pred(
    img_langualge_model: ImgLanguageModel,
    img_file_paths: List[str],
    device: torch.device,
):
    img_pils = []
    captions = []
    for img_file_path in img_file_paths:
        img_pil, caption = create_caption_from_img_file(
            img_langualge_model=img_langualge_model,
            img_file_path=img_file_path,
            device=device,
        )
        img_pils.append(img_pil)
        captions.append(caption)

    num_rows = 1
    num_cols = len(img_pils)
    fig, axs = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(16, 9), squeeze=False
    )
    row_idx = 0
    for col_idx, img_pil in enumerate(img_pils):
        caption = captions[col_idx]
        ax = axs[row_idx, col_idx]
        ax.imshow(np.asarray(img_pil))
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        multiline_caption = None
        MAX_LINE_LEN = 20
        for i in range(0, len(caption), MAX_LINE_LEN):
            if multiline_caption is None:
                multiline_caption = f"{caption[i:i+MAX_LINE_LEN]}\n"
            else:
                multiline_caption = f"{multiline_caption}{caption[i:i+MAX_LINE_LEN]}\n"

        ax.set_title(
            multiline_caption,
            color="green",
            fontsize=8,
        )

    plt.tight_layout()

    return fig
