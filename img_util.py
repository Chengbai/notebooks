import matplotlib.pyplot as plt
import torch

from config import Config
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2

# from torchvision import transforms


try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def load_img_tensor(config: Config, img_file_path: Path) -> torch.tensor:
    # Load image from file
    img = Image.open(img_file_path)  # .convert("RGB")

    # Transforming and augmenting images
    img_transform = v2.Compose(
        [
            _convert_image_to_rgb,
            v2.ToImage(),
            v2.ToDtype(
                torch.uint8, scale=True
            ),  # optional, most input are already uint8 at this point
            v2.RandomResizedCrop(
                size=(config.img_h_size, config.img_w_size),
                scale=(0.6, 1.0),
                antialias=True,
            ),
            v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
            # v2.ColorJitter(brightness=0.5, hue=0.3),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    # augmentation 1
    img_aug_tensor1 = img_transform(img)

    # augmentation 2
    img_aug_tensor2 = img_transform(img)

    # If original image is a GrayScale, simplly do R=G=B=GrayScale, otherewise, keep as is.
    img_aug_tensor1 = img_aug_tensor1.expand(3, -1, -1)
    img_aug_tensor2 = img_aug_tensor2.expand(3, -1, -1)

    return img_aug_tensor1, img_aug_tensor2


def inverse_img_aug(img: torch.tensor) -> torch.tensor:
    img = img.cpu().permute(1, 2, 0)
    img = img * torch.tensor([0.26862954, 0.26130258, 0.27577711])
    img = img + torch.tensor([0.48145466, 0.4578275, 0.40821073])
    # plt.imshow(test_img)
    return img.permute(2, 0, 1)


def show_img_tensor_CHW(img_tensor: torch.tensor):
    plt.imshow(img_tensor.permute(1, 2, 0))  # C x H x W => H x W x C
