from arxiv_bert.data.preprocess import ArxivBertPreprocessor
from django_project import settings
from helpers import constants

import tensorflow as tf
from transformers import TFBertForSequenceClassification


class ArxivBertPredictor:
    model = TFBertForSequenceClassification.from_pretrained(
        settings.BASE_DIR / "static" / constants.FINETUNED_MODEL
    )
    preprocessor = ArxivBertPreprocessor()

    def predict(self, text):
        # Tokenize the example text
        inputs = self.preprocessor.preprocess_text(text)

        # Make predictions
        outputs = self.model(
            inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        predicted_class_index = tf.argmax(outputs.logits, axis=1).numpy()[0]

        predicted_class_label = constants.LABELS[predicted_class_index]

        return predicted_class_label
