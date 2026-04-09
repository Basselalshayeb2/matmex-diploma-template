import re
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

CUR_DIR = Path(__file__).parent
DATASET = CUR_DIR / "out/teeth_classifier_dataset.csv"
MODEL_OUT = CUR_DIR / "out/teeth_gate.joblib"

def normalize_russian_letters(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text).strip()
    return text

class RuleFeatures(BaseEstimator, TransformerMixin):
    """
    Adds a few boolean features that reflect dental structure:
    - has_fdi_tooth: contains 2-digit tooth like 11..48
    - has_word_zub: contains 'зуб'
    - has_jaw: contains 'челюст'
    - has_range: contains 'со X по Y'
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for text in X:
            t = normalize_russian_letters(str(text))
            has_fdi = 1 if re.search(r"\b(1[1-8]|2[1-8]|3[1-8]|4[1-8])\b", t) else 0
            has_zub = 1 if "зуб" in t else 0
            has_jaw = 1 if "челюст" in t else 0
            has_range = 1 if re.search(r"\bсо\s+\d+\s+по\s+\d+\b", t) else 0
            feats.append([has_fdi, has_zub, has_jaw, has_range])
        return np.array(feats, dtype=np.float32)

def main():
    df = pd.read_csv(DATASET)
    print(df)
    df["text"] = df["text"].astype(str).map(normalize_russian_letters)
    X = df["text"].tolist()
    y = df["label"].astype(int).values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Char n-grams work best on noisy STT text
    char_vec = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=120000,
    )

    model = Pipeline([
        ("features", FeatureUnion([
            ("char", char_vec),
            ("rules", RuleFeatures()),
        ])),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    print("Confusion matrix:\n", confusion_matrix(yte, pred))
    print(classification_report(yte, pred, target_names=["not_teeth", "teeth"]))

    joblib.dump(model, MODEL_OUT)
    print("Saved:", MODEL_OUT.resolve())

    # show a few example probs
    probs = model.predict_proba(Xte)[:, 1]
    print("Prob examples:", np.round(probs[:10], 3))

if __name__ == "__main__":
    main()