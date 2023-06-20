import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from transformers import TFBertForSequenceClassification
import numpy as np

from arxiv_bert.helpers import constants


def model_builder(hp):
    model = TFBertForSequenceClassification.from_pretrained(constants.BASE_MODEL)

    hp_learning_rate = hp.Choice("learning_rate", values=[5e-5, 4e-5, 3e-5, 2e-5])
    optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy("accuracy")],
    )

    return model


def param_search(X_train_tokenized, y_train, X_val_tokenized, y_val):
    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=3,
        factor=2,
        directory="param_search",
        project_name="param_search",
    )

    stop_early = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5)

    tuner.search(
        X_train_tokenized["input_ids"],
        np.array(y_train),
        epochs=3,
        validation_data=(X_val_tokenized["input_ids"], np.array(y_val)),
        callbacks=[stop_early],
    )

    return tuner.get_best_hyperparameters(num_trials=1)[0]
