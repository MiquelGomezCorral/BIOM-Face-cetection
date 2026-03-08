from PIL import Image
import numpy as np
from tqdm import tqdm
from maikol_utils.file_utils import list_dir_files
from maikol_utils.print_utils import print_separator
import os
from src.config import Configuration

    
def create_partition(CONFIG: Configuration, rng):

    files, n = list_dir_files(CONFIG.all_path, recursive=True)
    files = set(files)
        
    test = set(rng.choice(list(files), size=int(n * CONFIG.test_split), replace=False))
    val = set(rng.choice(list(files - test), size=int(n * CONFIG.val_split), replace=False))
    train = files - test - val

    print(f"Train: {len(train)} images")
    print(f"Val: {len(val)} images")
    print(f"Test: {len(test)} images")

    def copy_as_jpg(src: str, dst_dir: str):
        base = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(dst_dir, base + ".jpg")
        with Image.open(src) as img:
            img.convert("RGB").save(dst, "JPEG")

    for split_paths, paths in zip([CONFIG.train_path, CONFIG.val_path, CONFIG.test_path], [train, val, test]):
        print_separator(split_paths)
        for path in tqdm(list(paths), desc=f"Copying {split_paths} images"):
            copy_as_jpg(path, split_paths)
