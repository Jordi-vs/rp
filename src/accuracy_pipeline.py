import os

import cv2
import fastwer

from panel_extractor.PanelExtractor import PanelExtractor
from text_extractor.OCRPreprocessor import crop_image_file
from text_extractor.TextExtractor import extract_text


def calculate_wer_cer_accuracy(ocr_output, ground_truth):
    common_keys = set(ocr_output.keys()) & set(ground_truth.keys())
    wer, cer = 0.0, 0.0
    for key in common_keys:
        ocr_value = re.sub(r'[^a-zA-Z0-9\s]', '',
                           str(ocr_output[key]).lower())  # Remove non-alphanumeric characters except whitespace
        gt_value = re.sub(r'[^a-zA-Z0-9\s]', '',
                          str(ground_truth[key]).lower())  # Remove non-alphanumeric characters except whitespace
        # print(ocr_value)
        # print(gt_value)
        # print(fastwer.score_sent(ocr_value, gt_value, char_level=False))
        # print(fastwer.score_sent(ocr_value, gt_value, char_level=True))
        wer += fastwer.score_sent(ocr_value, gt_value, char_level=False)
        cer += fastwer.score_sent(ocr_value, gt_value, char_level=True)

    div = max(len(common_keys), 1)
    wer /= div
    cer /= div
    return wer, cer


def calculate_accuracy(ocr_output, ground_truth):
    common_keys = set(ocr_output.keys()) & set(ground_truth.keys())

    total_words = 0
    correct_words = 0
    total_characters = 0
    correct_characters = 0

    for key in common_keys:
        ocr_value = re.sub(r'\W+', '',
                           str(ocr_output[key]).lower())  # Remove non-alphanumeric characters from OCR value
        gt_value = re.sub(r'\W+', '',
                          str(ground_truth[key]).lower())  # Remove non-alphanumeric characters from ground truth value

        ocr_words = ocr_value.split()
        gt_words = gt_value.split()

        total_words += len(gt_words)
        correct_words += sum(ocr_word == gt_word for ocr_word, gt_word in zip(ocr_words, gt_words))

        total_characters += sum(len(word) for word in gt_words)
        correct_characters += sum(ocr_char == gt_char for ocr_word, gt_word in zip(ocr_words, gt_words)
                                  for ocr_char, gt_char in zip(ocr_word, gt_word))

    if total_words == 0 or total_characters == 0:
        return 0, 0

    word_accuracy = (correct_words / total_words) * 100
    character_accuracy = (correct_characters / total_characters) * 100

    return word_accuracy, character_accuracy


import re
from fuzzywuzzy import fuzz


def calculate_similarity(obj1, obj2):
    similarity_scores = {}

    # Get common keys from both objects
    common_keys = set(obj1.keys()) & set(obj2.keys())
    for key in common_keys:
        # value1 = re.sub(r'[^a-zA-Z0-9\s]', '', str(obj1[key]).lower())  # Remove non-alphanumeric characters except whitespace
        # value2 = re.sub(r'[^a-zA-Z0-9\s]', '', str(obj2[key]).lower())  # Remove non-alphanumeric characters except whitespace
        value1 = re.sub(r'\W+', '', str(obj1[key]).lower())  # Remove all non-alphanumeric characters
        value2 = re.sub(r'\W+', '', str(obj2[key]).lower())  # Remove all non-alphanumeric characters
        # Calculate string similarity using fuzz.ratio or any other metric you prefer
        similarity_score = fuzz.ratio(value1, value2)

        similarity_scores[key] = similarity_score

    return similarity_scores


def extract_csv_information(csv_file):
    result = {}

    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)

        for row in reader:
            original_filename = row['original_filename']
            comics_text_box = row['Comics_text_box']

            result[original_filename] = comics_text_box

    return result


def concatenate_dictionary_values(dictionary):
    result = {}

    for key, value in dictionary.items():
        key_split = key.split('/')[1]
        parts = key_split.split('_')[0:3]  # Extract the necessary parts for prefix
        prefix = '_'.join(parts).split('.')[0] + '.png'

        # Check if there's a matching prefix in the dictionary
        if prefix in result:
            result[prefix] += ' ' + value
        else:
            result[prefix] = value

    return result


import csv


def get_original_filenames(csv_path, dr):
    original_filenames = []

    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            original_filename = row['original_filename']
            original_filenames.append(dr + original_filename)

    return original_filenames


def accuracy_pipeline_2():  # get path to every file in the directory

    csv_path = f"../comics_data/Dilbert_test_data.csv"
    dr = f"C:/Users/jordi/comic-dcp/src/splitted-images/"
    full_images_paths = get_original_filenames(csv_path, dr)

    splitted_images_directory = "splitted-images"
    panel_extractor = PanelExtractor()
    splitted_images_paths = panel_extractor.extract_and_save_panels(full_images_paths, splitted_images_directory)

    # get transcriptions
    transcriptions = extract_text(splitted_images_paths, "vision-api", rescale=True, clustering=True, text_panel=True)
    concatenated_transcriptions = concatenate_dictionary_values(transcriptions)
    ground_truth = extract_csv_information("../comics_data/Dilbert_test_data.csv")
    similarity_scores = calculate_similarity(concatenated_transcriptions, ground_truth)
    for key, score in similarity_scores.items():
        print(f"{key}: {score}")
        print(f"Ground truth: {ground_truth[key].lower()}")
        print(f"Extracted text: {concatenated_transcriptions[key]}")
    # wer, cer = calculate_accuracy(concatenated_transcriptions, ground_truth)
    wer, cer = calculate_wer_cer_accuracy(concatenated_transcriptions, ground_truth)
    print(f"Word Accuracy (WER): {round(100 - wer, 2)}%")
    print(f"Character Accuracy (CER): {round(100 - cer, 2)}%")
    print(
        f"Latent Dirichlet allocation Accuracy (LDA): {round(sum(similarity_scores.values()) / len(similarity_scores), 2)}%")

def accuracy_pipeline_1():
    csv_path = f"../comics_data/Dilbert_test_data.csv"
    dr = f"C:/Users/jordi/comic-dcp/src/splitted-images/"
    full_images_paths = get_original_filenames(csv_path, dr)

    # full_images_paths = []

    # for dir, subdir, files in os.walk(f"../comics_data/dilbert_comics_png"):
    #     for file in files:
    #         p = os.path.join(dir, file)
    #         full_images_paths.append(p.replace("\\", "/"))
    # extract panels from each full illustration
    splitted_images_directory = "splitted-images"
    panel_extractor = PanelExtractor()
    splitted_images_paths = panel_extractor.extract_and_save_panels(full_images_paths, splitted_images_directory)

    cropped_paths = []
    if True:
        # iterate over all splitted images
        for path in splitted_images_paths:
            images = crop_image_file(path, 0.8)
            # save cropped images
            for i, image in enumerate(images):
                output_filename = f"{os.path.splitext(os.path.basename(path))[0]}_{i}.png"
                output_path = os.path.join("cropped_text", output_filename)
                cv2.imwrite(output_path, image)
                cropped_paths.append(output_path.replace("\\", "/"))

    # get transcriptions
    transcriptions = extract_text(cropped_paths, "vision-api", rescale=True, clustering=True, text_panel=True)
    concatenated_transcriptions = concatenate_dictionary_values(transcriptions)
    ground_truth = extract_csv_information("../comics_data/Dilbert_test_data.csv")
    similarity_scores = calculate_similarity(concatenated_transcriptions, ground_truth)
    for key, score in similarity_scores.items():
        print(f"{key}: {score}")
        print(f"Ground truth: {ground_truth[key].lower()}")
        print(f"Extracted text: {concatenated_transcriptions[key]}")
    # wer, cer = calculate_accuracy(concatenated_transcriptions, ground_truth)
    wer, cer = calculate_wer_cer_accuracy(concatenated_transcriptions, ground_truth)
    print(f"Word Accuracy (WER): {round(100 - wer, 2)}%")
    print(f"Character Accuracy (CER): {round(100 - cer, 2)}%")
    print(f"Latent Dirichlet allocation Accuracy (LDA): {round(sum(similarity_scores.values())/len(similarity_scores),2)}%")
# Example usage

if __name__ == "__main__":
    # get path to every file in the directory
    accuracy_pipeline_2()
    # accuracy_pipeline_1()


    # visualize
    # for path in splitted_images_paths:
    #     image = cv2.imread(path)
    #     cv2.imshow("path", image)
    #     print(concatenated_transcriptions[path])
    #     cv2.waitKey(0)
