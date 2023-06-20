# arxiv-bert-classification
This repository contains a trainable BERT model experiment on Kaggle Arxiv Dataset [https://www.kaggle.com/datasets/Cornell-University/arxiv]
It also has a Django REST Framework app that runs a server and provides an API for predicting on an already trained model

I recommend going through the EDA analysis part first -> [https://github.com/RosenKrumov/arxiv-bert-classification/blob/main/arxiv_bert/analysis/eda-dataset-creation.ipynb]

## Pre-trained model scores on unseen test set of 187 samples:
Accuracy: 0.79 <br>
Validation accuracy: 0.77 <br>
Averaged Macro Precision: 0.62 <br>
Averaged Macro Recall: 0.60 <br>
Averaged F1 Score: 0.61

## API reference:
GET /api/prediction - to go to the predictions page
POST /api/prediction - to play with the model

## Steps to start the server and run predictions:
1) Download the model from this link - https://drive.google.com/file/d/11GUA5Exsu8fQ907ozY-JKTEoK5Tx6vod/view?usp=drive_link and store it in the `static` folder
2) Unzip the downloaded model
3) You can optionally create a virtual environment for cleaner package management - `python3 -m venv .venv` and then `source .venv/bin/activate`
4) Install the requirements with the following command - `python3 -m pip install -r requirements.txt`
5) Export the `PYTHONPATH` variable with the following command - `EXPORT PYTHONPATH=<path_to_repo>/arxiv_bert`
6) Run the following command to start the server - `python3 manage.py runserver`
7) Browse @ localhost:8000/api/prediction and have fun!
