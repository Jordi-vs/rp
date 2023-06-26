import base64
import io
import textwrap
from collections import Counter

import numpy as np
import pytesseract as tess
from google.cloud import aiplatform
from google.cloud import vision
from google.cloud.aiplatform.gapic.schema import predict


def predict_image_object_detection_sample(
        project: str,
        endpoint_id: str,
        filename: str,
        location: str = "us-central1",
        api_endpoint: str = "europe-west4-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    with open(filename, "rb") as f:
        file_content = f.read()

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageObjectDetectionPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_object_detection_1.0.0.yaml for the format of the parameters.
    parameters = predict.params.ImageObjectDetectionPredictionParams(
        confidence_threshold=0.5, max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/image_object_detection_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))
    return predictions


import cv2

def get_image_dimensions(file: str):
    image = cv2.imread(file)
    height, width, _ = image.shape
    return width, height

def detect_text_blocks_google(path: str):
    response = predict_image_object_detection_sample(
        project="497463487716",
        endpoint_id="8942521582748696576",
        location="europe-west4",
        filename=path
    )
    result = []
    image_width, image_height = get_image_dimensions(path)  # Assuming a function named get_image_dimensions to retrieve the image dimensions

    for prediction in response:
        for predict in prediction['bboxes']:
            x1 = int(predict[0] * image_width)
            x2 = int(predict[1] * image_width)
            y1 = int(predict[2] * image_height)
            y2 = int(predict[3] * image_height)
            result.append((x1, y1, x2, y2))

    # Sort the results in ascending order by x1
    result.sort(key=lambda bbox: bbox[0])

    # Sort the results within 10 difference in x1 based on ascending y1
    final_result = []
    i = 0
    while i < len(result):
        curr_bbox = result[i]
        final_result.append(curr_bbox)
        j = i + 1
        while j < len(result) and result[j][0] - curr_bbox[0] <= 50:
            final_result.append(result[j])
            j += 1
        final_result[i:j] = sorted(final_result[i:j], key=lambda bbox: bbox[1])
        i = j

    return final_result




def detect_text_blocks(
        path: str, block_confidence: float
) -> list[tuple[int, int, int, int]]:
    """Detect text blocks in the image found on the given path. Detected text
    blocks have a confidence greater than the given block confidence.

    :param path: A string indicating the path to the image.
    :return: A list of 4-tuples (x_min, y_min, x_max, y_max) representing the
        bounding boxes of the detected blocks.
    :param block_confidence: A float indicating the minimum confidence for the
        detected blocks.
    """
    # Run detection
    client = vision.ImageAnnotatorClient()
    with io.open(path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(
                response.error.message
            )
        )

    # Identify the blocks with high confidence
    blocks_vertices = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            if block.confidence < block_confidence:
                continue
            blocks_vertices.append(block.bounding_box)

    # Convert the vertices to 4-tuples
    bboxes = []
    for bv in blocks_vertices:
        pts = np.array([[vertex.x, vertex.y] for vertex in bv.vertices], np.int32)
        xmin = np.min([x[0] for x in pts])
        ymin = np.min([x[1] for x in pts])
        xmax = np.max([x[0] for x in pts])
        ymax = np.max([x[1] for x in pts])
        bboxes.append((xmin, ymin, xmax, ymax))

    return bboxes


def crop_image_file(file, model, confidence=0.8):
    image = cv2.imread(file)

    if model == "google":
        bounding_boxes = detect_text_blocks_google(file)
    else:
        bounding_boxes = detect_text_blocks(file, confidence)
    cropped_images = []

    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox

        # Crop the region of interest within the bounding box
        cropped_image = image[y1:y2, x1:x2]
        cropped_images.append(cropped_image)
    return cropped_images

def replace_text_in_image(file, new_text, confidence=0.8, font_size=0.6):
    image = cv2.imread(file)
    bounding_boxes = detect_text_blocks(file, confidence)
    # bounding_boxes = [(340, 20, 580, 100)]
    modified_images = []

    for index, bbox in enumerate(bounding_boxes):
        x1, y1, x2, y2 = bbox

        # Get the region of interest within the bounding box
        roi = image[y1:y2, x1:x2]

        # Calculate the most occurring color within the bounding box
        colors, counts = zip(*Counter([tuple(color) for color in roi.reshape(-1, 3)]).items())
        most_common_color = tuple(int(color) for color in colors[np.argmax(counts)])

        # Add a new box containing text in place of the removed bounding box
        if index >= len(new_text):
            text = ""
        else:
            text = new_text[index]
        text_color = (0, 0, 0)  # Red text color, adjust color if needed
        text_thickness = 1

        # Calculate the maximum allowed width for the wrapped text
        max_text_width = x2 - x1  # Adjust the padding as needed

        # Calculate the font-specific size
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_size, text_thickness)[0]
        line_height = text_size[1] + 5  # Adjust the line height as needed

        # Wrap the text to fit within the rectangle width
        if len(text) > 0:
            wrapped_text = textwrap.wrap(text, width=int(max_text_width / (text_size[0] / len(text))))
        else:
            wrapped_text = []

        # Calculate the position for the new box based on the text size and original bounding box
        text_x = x1  # Adjust the padding as needed
        text_y = y1

        # Draw the new box and wrapped text on the modified image
        cv2.rectangle(image, (text_x, text_y), (x2, y2),
                      most_common_color, -1)  # Use most occurring color

        for line in wrapped_text:
            cv2.putText(image, line, (text_x, text_y + 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color,
                        text_thickness)
            text_y += line_height

        modified_images.append(image)

        # Display the modified images
    # for modified_image in modified_images:
    #     cv2.imshow("Modified Image", modified_image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #
    cv2.imshow("Modified Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image


def rescale_for_ocr(image, tesseract_path, font_size=30):
    """
    Performs automatic rescaling for OCR - performs an intiial OCR pass, determines the current letter height,
    and rescales the image to match desired letter height. :param image: image array to rescale. :param
    tesseract_path: the path to the Tesseract OCR. :param font_size: the desired letter height in the output image.
    :return: The rescaled image.
    """
    tess.pytesseract.tesseract_cmd = tesseract_path
    orig_image = image.copy()
    image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    heights = []
    custom_config = r"-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-"
    boxes = tess.image_to_boxes(image, config=custom_config)
    h, w, _ = image.shape
    for b in boxes.splitlines():
        b = b.split()
        if len(b) == 6:
            y1, y2 = int(b[2]), int(b[4])
            heights.append(y2 - y1)

    if len(heights) < 5:
        return image

    scale = min(8, 3 * font_size / np.median(heights))
    rescaled_image = cv2.resize(orig_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return rescaled_image


def binarize_for_ocr(image):
    """
    Binarizes the image using adaptive thresholding.
    :param image: The image to process.
    :return: The binarized image.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binarized = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    binarized = cv2.fastNlMeansDenoising(binarized, None, 10, 10, 10)
    return binarized
