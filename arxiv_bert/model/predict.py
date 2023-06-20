import tensorflow as tf
import numpy as np

from arxiv_bert.data.preprocess import ArxivBertPreprocessor
from arxiv_bert.helpers import constants
from django_project import settings


class ArxivBertPredictor:
    model = tf.keras.models.load_model(
        settings.BASE_DIR / "static" / constants.FINETUNED_MODEL
    )
    preprocessor = ArxivBertPreprocessor()

    # Return raw predictions, used by the evaluator
    def get_raw_prediction(self, text):
        # Tokenize the example text
        inputs = self.preprocessor.preprocess_text(text)
        # Make predictions
        outputs = self.model(
            {
                "input_ids": np.asarray(inputs["input_ids"]),
                "token_type_ids": np.asarray(inputs["token_type_ids"]),
                "attention_mask": np.asarray(inputs["attention_mask"]),
            }
        )

        return outputs

    # Return predictions in user-readable format, used by the API
    def predict(self, text):
        outputs = self.get_raw_prediction(text)
        predicted_class_index = tf.argmax(outputs["logits"], axis=1).numpy()[0]
        predicted_class_label = constants.LABELS[predicted_class_index]
        return predicted_class_label
