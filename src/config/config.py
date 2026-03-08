import os
from maikol_utils.file_utils import make_dirs
from dataclasses import dataclass

@dataclass
class Configuration:
    """Configuration dataclass to hold application settings."""
    # =========================
    seed: int = 42

    # =========================
    DATA_PATH: str = os.path.join("data")
    wider_path: str = os.path.join("data", "WIDER_train", "images")
    dataset_path: str = os.path.join("data", "Dataset")
    all_path: str = os.path.join(dataset_path, "all")
    train_path: str = os.path.join(dataset_path, "img_train")
    train_f_path: str = os.path.join(dataset_path, "img_train_f")
    val_path: str = os.path.join(dataset_path, "img_val")
    val_f_path: str = os.path.join(dataset_path, "img_val_f")
    test_path: str = os.path.join(dataset_path, "img_test")
    test_f_path: str = os.path.join(dataset_path, "img_test_f")

    MODELS_PATH: str = os.path.join("models")
    best_cnn_model_path: str = os.path.join(MODELS_PATH, "best_cnn_model.ckpt")
    best_ori_model_path: str = os.path.join(MODELS_PATH, "best_ori_model.ckpt")

    val_split: float = 0.1
    test_split: float = 0.1

    aug_prob: float = 0.65

    max_files: int = 100
    gray_scale: bool = True
    

    # =========================
    crop_size: int = 256
    stride: int = 4
    subsample_factor: float = 0.8
    normalize_window: int = 5

    # =========================
    batch_size: int = 32
    num_workers: int = 4

    # =========================
    lr: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 100

    def __post_init__(self):
        make_dirs([
            self.train_path, self.val_path, self.test_path, 
            self.train_f_path, self.val_f_path, self.test_f_path, 
            self.MODELS_PATH
        ])

