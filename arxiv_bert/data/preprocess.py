import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer
from sklearn import preprocessing
import tensorflow as tf

import re, string
from tqdm import tqdm

from arxiv_bert.helpers import constants

tqdm.pandas()

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


class ArxivBertPreprocessor:
    label_encoder = preprocessing.LabelEncoder()
    stop = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokenizer = BertTokenizer.from_pretrained(constants.BASE_MODEL)

    def text_clean(self, x):
        # first we lowercase everything
        x = x.lower()
        x = " ".join([word for word in x.split(" ") if word not in self.stop])

        # remove unicode characters
        x = x.encode("ascii", "ignore").decode()
        x = re.sub(r"https*\S+", " ", x)
        x = re.sub(r"http*\S+", " ", x)

        # then use regex to remove @ symbols and hashtags
        x = re.sub(r"@\S", "", x)
        x = re.sub(r"#\S+", " ", x)
        x = re.sub(r"\'\w+", "", x)
        x = re.sub("[%s]" % re.escape(string.punctuation), " ", x)
        x = re.sub(r"\w*\d+\w*", "", x)
        x = re.sub(r"\s{2,}", " ", x)
        x = re.sub(r"\s[^\w\s]\s", "", x)

        # remove single letters and numbers surrounded by space
        x = re.sub(r"\s[a-z]\s|\s[0-9]\s", " ", x)

        tokens = word_tokenize(x)
        lemmatized = [self.lemmatizer.lemmatize(word) for word in tokens]
        x = " ".join(lemmatized)

        return x

    def preprocess_text(self, text):
        return self.tokenizer(
            text,
            return_tensors="np",
            max_length=constants.MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )

    def encode_labels(self, labels):
        return self.label_encoder.fit_transform(labels)

    def prepare_dataset(self, X, y):
        X_tokenized = self.tokenizer(
            X,
            return_tensors="np",
            max_length=constants.MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )

        y_encoded = self.label_encoder.fit_transform(y)
        ds = tf.data.Dataset.from_tensor_slices((dict(X_tokenized), y_encoded)).batch(
            constants.BATCH_SIZE
        )

        return ds, X_tokenized, y_encoded
