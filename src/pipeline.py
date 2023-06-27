import argparse
import csv
import os

import cv2

from accuracy_pipeline import calculate_similarity, extract_csv_information, calculate_wer_cer_accuracy
from panel_extractor.PanelExtractor import PanelExtractor
from text_extractor.OCRPreprocessor import crop_image_file
from text_extractor.TextExtractor import extract_text


def main(args):
    segmented_panels_paths = []
    if (not args.use_existing_panels):
        full_images_paths = []
        for dir, subdir, files in os.walk(args.comics_path):
            for file in files:
                p = os.path.join(dir, file)
                full_images_paths.append(p.replace("\\", "/"))

        if not os.path.exists(args.storage_for_segmented_panels):
            os.makedirs(args.storage_for_segmented_panels)
        segmented_panels_directory = args.storage_for_segmented_panels
        panel_extractor = PanelExtractor()
        # 136: 150
        segmented_panels_paths = panel_extractor.extract_and_save_panels(full_images_paths[:3],
                                                                         segmented_panels_directory)

    else:
        for dir, subdir, files in os.walk(args.comics_path):
            for file in files:
                p = os.path.join(dir, file)
                segmented_panels_paths.append(p.replace("\\", "/"))

    cropped_paths = []
    if args.text_boxes != 'none':
        if not os.path.exists(args.storage_for_cropped_text):
            os.makedirs(args.storage_for_cropped_text)
        # iterate over all splitted images
        for path in segmented_panels_paths:
            images = crop_image_file(path, args.ocr_model, 0.8)
            # save cropped images
            for i, image in enumerate(images):
                output_filename = f"{os.path.splitext(os.path.basename(path))[0]}_{i}.png"
                output_path = os.path.join(args.storage_for_cropped_text, output_filename)
                cv2.imwrite(output_path, image)
                cropped_paths.append(output_path)
    else:
        cropped_paths = segmented_panels_paths
    # get transcriptions
    transcriptions = extract_text(cropped_paths, "vision-api", clustering=True, text_panel=True)

    # Write data to CSV file
    if args.csv_path is None:
        with open(args.csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Image', 'Text'])  # Write header row

            for image_path, text in transcriptions.items():
                writer.writerow([image_path, text])

    if args.accuracy_path is not None:
        concatenated_transcriptions = get_names(transcriptions)
        # ground_truth = extract_csv_information("../comics_data/Dilbert_test_data.csv")
        ground_truth = extract_csv_information(args.accuracy_path)
        similarity_scores = calculate_similarity(concatenated_transcriptions, ground_truth)
        for key, score in similarity_scores.items():
            print(f"{key}: {score}")
            print(f"Ground truth: {ground_truth[key].lower()}")
            print(f"Extracted text: {concatenated_transcriptions[key]}")
        wer, cer = calculate_wer_cer_accuracy(concatenated_transcriptions, ground_truth)
        print(f"Word Accuracy (WER): {round(100 - wer, 2)}%")
        print(f"Character Accuracy (CER): {round(100 - cer, 2)}%")
        print(
            f"Latent Dirichlet allocation Accuracy (LDA): {round(sum(similarity_scores.values()) / max(len(similarity_scores), 1), 2)}%")

    # visualize
    if args.visualize:
        for path in cropped_paths:
            image = cv2.imread(path)
            cv2.imshow("path", image)
            print(transcriptions[path])
            cv2.waitKey(0)

def get_names(dictionary):
    result = {}
    for key, value in dictionary.items():
        key_split = key.split('/')[-1]
        result[key_split] = value

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

if __name__ == "__main__":
    # get path to every file in the directory
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('ocr_model', type=str,
                        help='OCR model to use:\nvision-api: Use Google Vision API\nTesseract: Use local Tesseract '
                             'model\n')
    parser.add_argument('text_boxes', type=str,
                        help='What text box location strategy to use:\nNone: No text boxes\nstandard: Use Google '
                             'Vision API to detect text boxes\nfine tuned: Use Google Vision API fined tuned model to '
                             'detect text boxes\n')
    parser.add_argument('comics_path', type=str,
                        help='Path to comics or path to panels if --use_existing_panels is set to True', default=None)
    parser.add_argument('--csv_path', '-c', type=str, help='Path to CSV file where you want to save the results')
    parser.add_argument('--use_existing_panels', '-up', type=bool,
                        help='Use panels that have already been previously segmented', default=False)
    parser.add_argument('--storage_for_segmented_panels', '-sp', type=str,
                        help='Path to directory where you want to save the segmented panels. Default = segmented-panels',
                        default="segmented-panels")
    parser.add_argument('--storage_for_cropped_text', '-st', type=str,
                        help='Path to directory where you want to save the cropped text. Default = cropped-text',
                        default="cropped-text")
    parser.add_argument('--visualize', '-v', type=bool, help='Visualize the results', default=False)
    parser.add_argument('--accuracy_path', '-a', type=str, help='Calculate accuracy', default=False)

    args = parser.parse_args()

    assert args.ocr_model in ['vision-api', 'tesseract'], "Invalid OCR model"
    assert args.text_boxes in ['none', 'standard', 'fine tuned'], "Invalid text box location strategy"
    main(args)

