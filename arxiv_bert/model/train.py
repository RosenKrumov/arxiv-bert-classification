import pickle
from transformers import TFBertForSequenceClassification
import tensorflow as tf

from django_project import settings
from arxiv_bert.data.preprocess import ArxivBertPreprocessor
from arxiv_bert.helpers import constants
from arxiv_bert.model.hyperparam_search import param_search

with open(settings.BASE_DIR / "arxiv_bert" / "data" / "trainset.pickle", "rb") as file:
    trainset = pickle.load(file)

preprocessor = ArxivBertPreprocessor()

train_ds, X_train_tokenized, y_train_encoded = preprocessor.prepare_dataset(
    trainset["X_train"], trainset["y_train"]
)
val_ds, X_val_tokenized, y_val_encoded = preprocessor.prepare_dataset(
    trainset["X_val"], trainset["y_val"]
)

model = TFBertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=8
)

tuned_params = param_search(
    X_train_tokenized, y_train_encoded, X_val_tokenized, y_val_encoded
)

loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(
    learning_rate=tuned_params.get("learning_rate"), epsilon=1e-08
)

model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

model.fit(train_ds, epochs=4, validation_data=val_ds)

model.save(settings.BASE_DIR / "static" / constants.FINETUNED_MODEL)
