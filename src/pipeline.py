import os

import cv2

from panel_extractor.PanelExtractor import PanelExtractor
from text_extractor.TextExtractor import extract_text

if __name__ == "__main__":
    # get path to every file in the directory
    full_images_paths = []
    for dir, subdir, files in os.walk(f"../comics_data/dilbert_comics_png"):
        for file in files:
            p = os.path.join(dir, file)
            full_images_paths.append(p.replace("\\", "/"))
    # extract panels from each full illustration
    splitted_images_directory = "splitted-images"
    panel_extractor = PanelExtractor()
    splitted_images_paths = panel_extractor.extract_and_save_panels(full_images_paths[:3], splitted_images_directory)

    # get transcriptions
    transcriptions = extract_text(splitted_images_paths, "tesseract", clustering=True)

    # visualize
    for path in splitted_images_paths:
        image = cv2.imread(path)
        cv2.imshow("path", image)
        print(transcriptions[path])
        cv2.waitKey(0)
