# Pipeline construction for the automated text retrieval, editing, and deletion in comic illustrations

### Context
This repository is forked from the [Automated Comic Dataset Construction Pipeline](https://github.com/mstyczen/comic-dcp), created by Maciej Styczen which contained the source code for his Bachelor's thesis on the topic of "Automated Comic Dataset Construction Pipeline" at TU Delft in 2021.

The research paper which provides detailed explanations and motivation behind the many choices made throughout this project and the evaluation on the performance of this pipeline can be found on the  [TU Delft Repository](https://repository.tudelft.nl/islandora/object/uuid%3A4436f83c-7aad-401e-91b9-99cb157ee585).

Thank you to prof. Lydia Chen and dr. Zilong Zhao for providing me with assistance throughout the entirety of the project.

### Goal of the system
With the rapid acceleration of Machine Learning and AI in today's world, the demand for high quality data is increasing significantly. Machine Learning models are often dictated by the quality of the data that they are provided with, and as such the quote of "garbage in garbage out". Currently, there is a severe lack of high quality data to train on, often due to the need to manually label large parts of the data. This results in major bottlenecks for the further improvement of models.

The goal of this repository is to alleviate this bottleneck by providing a novel source of high quality data through the process of extracting text from comic strips.

On top of this, the ability to modify or delete the text inside of the comics gives way to another source of data that can be valuable to users. A primary example of this is for generative AI, which prefers to have textless images.

### Usability

In order to begin running this project, you must either have [Tesseract](https://tesseract-ocr.github.io/tessdoc/), or [Vision API setup](https://cloud.google.com/vision/docs/setup) setup on your machine.
When done we advise creating a new virtual environment using 
```
python -m venv /path/to/new/virtual/environment
```
then 
```
pip install pybind11
```
and finally
```
pip install -r requirements.txt
```

To run an example, we can run 
```
python.exe .\pipeline.py tesseract standard ..\comics_data\dilbert_comics_png\ results.csv
```

#### Command Line Arguments

```positional arguments:
  ocr_model             OCR model to use:
                        vision-api: Use Google Vision API
                        tesseract: Use local Tesseract model
  text_boxes            What text box location strategy to use:
                        none: No text boxes 
                        standard: Use Google Vision API to detect text boxes 
                        finetuned: Use Google Vision API fined tuned model to detect text boxes
  comics_path           Path to comics OR path to panels if --use_existing_panels is set to True
  csv_path              Path to CSV file where you want to save the results

options:
  --use_existing_panels, -up: Boolean
                        Use panels that have already been previously segmented
  --storage_for_segmented_panels, -sp: String
                        Path to directory where you want to save the segmented panels.
  --storage_for_cropped_text, -st: String
                        Path to directory where you want to save the cropped text. Default = cropped-text
  --visualize, -v: Boolean 
                        Visualize the results
  --accuracy_path, -a: String
                        Provide optional path to ground truth data to get accuracy results
```

### Reproduce finetuning

Finetuning is done through Google Vertex AI, with [details here](https://cloud.google.com/vertex-ai/docs/tabular-data/tabular-workflows/e2e-automl). Our steps were to use the manual labelling ability offered by Vertex AI for almost 500 comic panels to label every single text box. After this, the training feature was utilized in order to finetune the model on text boxes. Finally, we can deploy the model to an endpoint as well using Vertex AI. With the endpoint we can make requests to the model and receive bounding box locations in the response, with which we will carry out the rest of our pipeline.