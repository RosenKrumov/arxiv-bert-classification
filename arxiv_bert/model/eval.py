from cgi import test
import tensorflow as tf
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from arxiv_bert.data.preprocess import ArxivBertPreprocessor

from arxiv_bert.model.predict import ArxivBertPredictor

from django_project import settings

with open(settings.BASE_DIR / "data" / "testset.pickle", "rb") as file:
    testset = pickle.load(file)

X_test = testset["X_test"]

predictor = ArxivBertPredictor()
preprocessor = ArxivBertPreprocessor()

y_test = preprocessor.encode_labels(test["y_test"])

# Make predictions
outputs = [predictor.predict(x) for x in X_test]
y_pred = [tf.argmax(output.logits, axis=1).numpy()[0] for output in outputs]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score = (2 * precision * recall) / (precision + recall)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1_score}")
