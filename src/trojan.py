"""Converts the images found in the input folder to png format. The comics in each year
are saved in chronological order.
"""

import os

from PIL import Image
from tqdm import tqdm

DATA_DIR = "comics_data"

input_folder_path = f"{DATA_DIR}/dilbert_comics_gif"
output_folder_path = f"{DATA_DIR}/dilbert_comics_png"

if not os.path.exists(input_folder_path):
    raise ValueError(f"Could not find {input_folder_path}.")

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for year in tqdm(sorted(os.listdir(input_folder_path)), desc="Years"):
    input_year_dir = f"{input_folder_path}/{year}"
    output_year_dir = f"{output_folder_path}/{year}"
    os.makedirs(output_year_dir)
    for idx, image_file in enumerate(sorted(os.listdir(f"{input_year_dir}"))):
        im = Image.open(f"{input_year_dir}/{image_file}")
        im.save(f"{output_year_dir}/{year}_{idx}.png", "PNG")
