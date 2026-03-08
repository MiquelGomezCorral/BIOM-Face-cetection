import numpy as np

from src.config import Configuration


def local_normalize_image(CONFIG: Configuration, img: np.ndarray):
    integral = get_integral_image(img) 
    integral_2 = get_integral_squared_image(img)

    x_idx, y_idx = np.indices(img.shape)
    vect_normalize = np.vectorize(
        local_normalize_pixel,
        excluded=["integral", "integral_2", "win_size"]
    )

    return vect_normalize(
        img,
        integral=integral,
        integral_2=integral_2,
        x=x_idx,
        y=y_idx,
        win_size=CONFIG.normalize_window
    )

def local_normalize_pixel(pixel, integral, integral_2, x, y, win_size):
    win = win_size // 2
    max_row = integral.shape[0] - 1
    max_col = integral.shape[1] - 1

    # Ajustar límites
    r1 = max(0, x - win)
    r2 = min(max_row, x + win)
    c1 = max(0, y - win)
    c2 = min(max_col, y + win)

    pixels = (r2 - r1 + 1) * (c2 - c1 + 1)

    suma = get_integral_sum(integral, r1, c1, r2, c2)
    suma_2 = get_integral_sum(integral_2, r1, c1, r2, c2)
    mu = suma / pixels

    var = max(0.0, (suma_2 - (2*mu*suma) + pixels*mu*mu)/pixels) 
    sig = np.sqrt(var)

    # Prevent division by zero
    if sig < 1e-6:
        return 0.0

    return (pixel - mu)/sig


# Versiones rápidas
def get_integral_image(img: np.ndarray) -> np.ndarray:
    return img.astype(np.int64).cumsum(axis=0).cumsum(axis=1)

def get_integral_squared_image(img: np.ndarray) -> np.ndarray:
    img2 = img.astype(np.int64) ** 2
    return img2.cumsum(axis=0).cumsum(axis=1)

def get_integral_sum(integral: np.ndarray, x1, y1, x2, y2) -> int:
    A = integral[x1 - 1, y1 - 1] if (x1 > 0 and y1 > 0) else 0
    B = integral[x1 - 1, y2] if x1 > 0 else 0
    C = integral[x2, y1 - 1] if y1 > 0 else 0
    D = integral[x2, y2]
    return D - B - C + A



# Versiones lentas

# def get_integral_image(img: np.ndarray):
#     integral = np.zeros_like(img, dtype=np.uint64) 

#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             integral[i, j] = img[0:i+1, 0:j+1].sum()

#     return integral

# def get_integral_squared_image(img: np.ndarray):
#     img_2 = np.square(img)
#     integral = np.zeros_like(img_2, dtype=np.uint64) 

#     for i in range(img_2.shape[0]):
#         for j in range(img_2.shape[1]):
#             integral[i, j] = img_2[0:i+1, 0:j+1].sum()

#     return integral
