from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import os
import joblib
from config import MODEL_FILE_NAME


def fit_model():
    data_dir = "movie_reviews"

    documents, labels = [], []

    for movie_id in os.listdir(data_dir):
        for file_name in os.listdir(os.path.join(data_dir, movie_id)):
            file_path = os.path.join(data_dir, movie_id, file_name)
            with open(file_path, encoding='utf8') as f:
                labels.append(next(f))
                documents.append(f.read())

    documents = documents[:1000]
    labels = labels[:1000]
    labels_binary = ["positive" if int(l) >= 7 else "negative" for l in labels]
    model = make_pipeline(CountVectorizer(), LogisticRegression())
    model.fit(documents, labels_binary)

    joblib.dump(model, MODEL_FILE_NAME)
