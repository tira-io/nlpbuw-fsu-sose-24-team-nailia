import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory


def train_lr():
    tira = Client()

    # Loading train data
    text_train = tira.pd.inputs("nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training")
    targets_train = tira.pd.truths("nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training")
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the training data and transform the text data into TF-IDF vectors
    X_train = vectorizer.fit_transform(text_train['text'])
    y_train = targets_train['generated']
    # Train model
    classifier = LogisticRegression() # Logistic Classifier with well interpretable results
    classifier.fit(X_train, y_train)
    return classifier, vectorizer
