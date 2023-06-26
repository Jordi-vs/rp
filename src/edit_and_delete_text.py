import argparse
import os

import cv2

from text_extractor.OCRPreprocessor import replace_text_in_image

new_text = [
    "",
    "",
    "",
    "",
    "",
    # "I am Dilbert and this is my comic strip and something is happening to dogbert and I am talking about it, there is a lot of text in this comic strip, I wonder what is going to happen next",
    # "I can't believe this is happening to me",
    # "I am Dogbert and for some reason I am a dog, which is weird because I am talking",
    # "I am Wally and I am lazy",
    # "Just incase you forgot, I am Dilbert",
    # "I am Dogbert and I am a dog",
]


def process_all_images(new_text):
    # Get the current directory
    base_path = os.getcwd()

    # Construct the paths for the input and output folders
    input_folder = os.path.join(base_path, "splitted-images2")
    output_folder = os.path.join(base_path, "blank_images_2022")

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all PNG files in the input folder
    input_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]

    for file_name in input_files:
        # Construct the full paths for each input and output image
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        # Load the image using cv2
        # image = cv2.imread(input_path)

        # Apply the text replacement function
        try:
            modified_image = replace_text_in_image(input_path, new_text)
            cv2.imwrite(output_path, modified_image)
        except:
            print("Error with image: ", input_path)
            continue
        # Save the modified image


# process_all_images(new_text)
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str, help='Path to the image file')
    parser.add_argument('-list_of_text', '-t', nargs='+', default="")

    args = parser.parse_args()

    text = ' '.join(args.list_of_text)

    new_text = text.split("/")

    replace_text_in_image(args.filepath, new_text)
