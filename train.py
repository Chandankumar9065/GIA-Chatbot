import json
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_model():
    with open("intents.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    texts = []
    labels = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            texts.append(pattern.lower())
            labels.append(intent["tag"])

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        lowercase=True
    )

    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    model.fit(X, labels)

    pickle.dump(model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    return model, vectorizer


if __name__ == "__main__":
    train_model()
    print("Model Trained Successfully")
