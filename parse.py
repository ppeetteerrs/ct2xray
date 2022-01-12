from glob import glob
from os import environ
from pathlib import Path
from random import sample

import cv2 as cv
import numpy as np
from tqdm.contrib.concurrent import process_map

output_dir = (Path(environ["DATA"]) / "chexpert_std")
output_dir.mkdir(exist_ok=True, parents=True)

files = glob("chexpert/train/patient*/study*/view1_frontal.jpg")
files = sample(files, 70000)
print(f"{len(files)} frontal CXR found.")
print(f"Saving outputs to {output_dir}")


def get_filename(img_path: str) -> str:
    patient_no = img_path.split("/patient")[1].split("/")[0]
    study_no = img_path.split("/study")[1].split("/")[0]
    view_no = img_path.split("/view")[1].split("_")[0]
    return f"{output_dir}/{patient_no}_{study_no}_{view_no}.png"


def read_img(img_path: str) -> np.ndarray:
    img = cv.imread(img_path)
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def remove_border(img: np.ndarray, tol=100) -> np.ndarray:
    mask = img > tol
    m, n = img.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def center_crop(img: np.ndarray) -> np.ndarray:
    x, y = img.shape
    dim = min(x, y)

    x_start = (x - dim) // 2
    y_start = (y - dim) // 2

    return img[x_start: x_start + dim, y_start: y_start + dim]


def process(img_path: str) -> str:
    img = read_img(img_path)
    img = remove_border(img)
    img = center_crop(img)
    img = cv.equalizeHist(img)
    img = cv.resize(img, (1024, 1024))
    if np.mean(img) < 50:
        return img_path
    else:
        cv.imwrite(get_filename(img_path), img)
        return "ok"


results = process_map(process, files, max_workers=4, chunksize=1)
okayed = sum([1 for item in results if item == "ok"])
print(f"{okayed}/{len(files)} images processed.")
