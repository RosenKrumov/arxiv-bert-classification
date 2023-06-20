import tensorflow as tf
import pickle
from tqdm import tqdm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from arxiv_bert.data.preprocess import ArxivBertPreprocessor
from arxiv_bert.model.predict import ArxivBertPredictor

from django_project import settings

# Load the testset
with open(settings.BASE_DIR / "arxiv_bert" / "data" / "testset.pickle", "rb") as file:
    testset = pickle.load(file)

predictor = ArxivBertPredictor()
preprocessor = ArxivBertPreprocessor()

# Preprocess X and y
X_test = testset["X_test"]
y_test = preprocessor.encode_labels(testset["y_test"])

# Make predictions
outputs = [predictor.get_raw_prediction(x) for x in tqdm(X_test)]
y_pred = [tf.argmax(output["logits"], axis=1).numpy()[0] for output in outputs]

# Calculate metrics
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1_score = (2 * precision * recall) / (precision + recall)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1_score}")
