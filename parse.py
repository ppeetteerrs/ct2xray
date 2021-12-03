from glob import glob

import numpy as np
from opencxr import load
from opencxr.algorithms import cxr_standardize
from opencxr.utils.file_io import write_file
from PIL import Image
from tqdm import tqdm

files = glob("chexpert/train/patient*/study*/*_frontal.jpg")[:10]
print(f"{len(files)} frontal CXR found.")
algo = load(cxr_standardize)


def new_filename(img_path: str) -> str:
    patient_no = img_path.split("/patient")[1].split("/")[0]
    study_no = img_path.split("/study")[1].split("/")[0]
    view_no = img_path.split("/view")[1].split("_")[0]
    return f"chexpert_std/{patient_no}_{study_no}_{view_no}.png"


def standardize(img_path: str) -> None:
    img = np.array(Image.open(img_path)).transpose()
    std_img, new_spacing, _ = algo.run(img, (1.0, 1.0))
    new_name = new_filename(img_path)
    write_file(new_name, std_img, new_spacing)


# res = process_map(standardize, files, max_workers=1)

for file in tqdm(files):
    standardize(file)
