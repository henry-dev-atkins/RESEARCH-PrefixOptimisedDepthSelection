from __future__ import annotations

from torchvision import transforms


_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD = (0.2470, 0.2435, 0.2616)


def build_image_transforms(dataset: str, train: bool):
    if dataset == "mnist":
        return transforms.ToTensor()
    if dataset in {"cifar10", "cifar100"}:
        ops = []
        if train:
            ops.extend(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
            ]
        )
        return transforms.Compose(ops)
    raise ValueError(f"Unsupported dataset: {dataset}")
