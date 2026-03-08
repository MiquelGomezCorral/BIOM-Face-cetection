import cv2
import numpy as np
from src.config import Configuration

def get_all_image_crops(CONFIG: Configuration, image_path: str):
    crops = []
    if CONFIG.gray_scale:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path)


    while img.shape[0] > CONFIG.crop_size and img.shape[1] > CONFIG.crop_size:
        crops.extend(get_image_crops(img, CONFIG.stride, CONFIG.crop_size))
        img = cv2.resize(
            img, 
            dsize=(
                int(img.shape[0] * CONFIG.subsample_factor), 
                int(img.shape[1] * CONFIG.subsample_factor)
            ), 
            interpolation=cv2.INTER_AREA
        )
    return crops

def get_image_crops(img: np.ndarray, stride: int, crop_size: int):
    h, w = img.shape[:2]
    crops = []
    for i in range(0, h - crop_size + 1, stride):
        for j in range(0, w - crop_size + 1, stride):
            crops.append(img[
                i:i + crop_size, 
                j:j + crop_size
            ])
    return crops
