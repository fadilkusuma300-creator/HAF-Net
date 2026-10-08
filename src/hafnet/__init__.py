"""HAF-Net package for imbalanced financial distress prediction."""

from .config import load_config
from .data import PreparedDataset, prepare_dataset

__all__ = ["HAFNet", "PreparedDataset", "load_config", "prepare_dataset"]


def __getattr__(name: str):
    if name == "HAFNet":
        from .model import HAFNet

        return HAFNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
