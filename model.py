import time
from functools import lru_cache
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

from storage import DATA_DIR, MODEL_DIR, load_metrics, save_metrics
from utils import build_combined_text, clean_text

DATASET_PATH = DATA_DIR / "dataset.csv"
MODEL_PATH = MODEL_DIR / "hybrid_model.joblib"
TRANSFORMER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
W_BERT = {"word": 0.40, "char": 0.25, "bert": 0.35}
W_PLAIN = {"word": 0.62, "char": 0.38}
LABELS = {"true": "real", "real": "real", "1": "real", "reliable": "real",
          "false": "fake", "fake": "fake", "0": "fake", "unreliable": "fake"}


# ---------------------------------------------------------------- data
def _read_normalised(path=DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    text_col = cols.get("text") or cols.get("article") or cols.get("content")
    label_col = cols.get("label") or cols.get("target") or cols.get("class")
    if text_col is None or label_col is None:
        raise ValueError("Dataset must contain at least 'text' and 'label' columns.")
    title_col = cols.get("title")
    src_col = cols.get("source") or cols.get("news_source") or cols.get("domain")
    out = pd.DataFrame({
        "title": df[title_col].fillna("").map(clean_text) if title_col else "",
        "text": df[text_col].fillna("").map(clean_text),
        "source": df[src_col].fillna("").astype(str).str.strip() if src_col else "",
        "label": df[label_col].astype(str).str.strip().str.lower().map(LABELS),
    })
    out = out.dropna(subset=["label"]).copy()
    out["combined_text"] = [build_combined_text(t, x) for t, x in zip(out["title"], out["text"])]
    return out[out["combined_text"].str.len() > 0].reset_index(drop=True)


def load_dataset(path=DATASET_PATH) -> pd.DataFrame:
    """Normalised dataset with exact duplicates removed (duplicates inflate every score)."""
    df = _read_normalised(path)
    raw_rows = len(df)
    df = df.drop_duplicates(subset="combined_text").reset_index(drop=True)
    df.attrs["raw_rows"], df.attrs["duplicates_removed"] = raw_rows, raw_rows - len(df)
    return df


def dataset_quality(path=DATASET_PATH) -> Dict:
    raw = _read_normalised(path)
    uniq = raw.drop_duplicates(subset="combined_text")
    counts = uniq["label"].value_counts().to_dict()
    warnings = []
    if len(uniq) < len(raw) * 0.8:
        warnings.append(f"{len(raw) - len(uniq)} of {len(raw)} rows are duplicates; only {len(uniq)} unique articles are used for training.")
    if len(uniq) < 1000:
        warnings.append(f"Only {len(uniq)} unique articles. That is far too few to generalise to real-world news; use the classifier as a weak signal only.")
    if counts and min(counts.values()) / max(counts.values()) < 0.6:
        warnings.append("Classes are imbalanced.")
    srcs = raw[raw["source"] != ""]
    purity = None
    if len(srcs):
        per_source = srcs.groupby(srcs["source"].str.lower())["label"].nunique()
        purity = float((srcs["source"].str.lower().map(per_source) == 1).mean())
        if purity > 0.95 and per_source.size >= 5:
            warnings.append("Source name almost perfectly predicts the label in this dataset. It is excluded from the text features so the model cannot just memorise outlets.")
    return {"rows": len(raw), "unique_articles": len(uniq), "label_counts": counts,
            "source_purity": purity, "warnings": warnings}


# ---------------------------------------------------------------- training
@lru_cache(maxsize=1)
def _get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(TRANSFORMER_MODEL_NAME)
    except Exception:
        return None


def _lr(C: float) -> LogisticRegression:
    return LogisticRegression(max_iter=4000, class_weight="balanced", C=C, solver="liblinear", random_state=42)


def _fit_branches(X: List[str], y, emb=None) -> Dict:
    word_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.95, max_features=30000, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=25000, sublinear_tf=True)
    word_clf = _lr(2.0).fit(word_vec.fit_transform(X), y)
    char_clf = _lr(1.5).fit(char_vec.fit_transform(X), y)
    bert_clf = _lr(2.0).fit(emb, y) if emb is not None else None
    return {"word_vec": word_vec, "word_clf": word_clf, "char_vec": char_vec, "char_clf": char_clf, "bert_clf": bert_clf}


def _p_fake(clf, X) -> np.ndarray:
    return clf.predict_proba(X)[:, list(clf.classes_).index("fake")]


def _ensemble(b: Dict, X: List[str], emb=None) -> np.ndarray:
    w = _p_fake(b["word_clf"], b["word_vec"].transform(X))
    c = _p_fake(b["char_clf"], b["char_vec"].transform(X))
    if b.get("bert_clf") is not None and emb is not None:
        return W_BERT["word"] * w + W_BERT["char"] * c + W_BERT["bert"] * _p_fake(b["bert_clf"], emb)
    return W_PLAIN["word"] * w + W_PLAIN["char"] * c


def train_model(dataset_path=DATASET_PATH, use_transformer: bool = True) -> Dict:
    df = load_dataset(dataset_path)
    counts = df["label"].value_counts()
    if len(df) < 20 or counts.min() < 5:
        raise ValueError("Need at least 20 unique articles with 5+ per class to train.")
    X, y = df["combined_text"].tolist(), df["label"].to_numpy()

    embedder = _get_embedder() if use_transformer else None
    emb = embedder.encode(X, batch_size=32, show_progress_bar=False) if embedder is not None else None

    # Honest evaluation: stratified cross-validation on de-duplicated data, not one tiny split.
    k = int(min(5, counts.min()))
    oof = np.zeros(len(df))
    fold_acc = []
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=42).split(X, y):
        b = _fit_branches([X[i] for i in tr], y[tr], emb[tr] if emb is not None else None)
        oof[te] = _ensemble(b, [X[i] for i in te], emb[te] if emb is not None else None)
        fold_acc.append(accuracy_score(y[te], np.where(oof[te] >= 0.5, "fake", "real")))

    pred = np.where(oof >= 0.5, "fake", "real")
    try:
        auc = round(float(roc_auc_score(y == "fake", oof)), 4)
    except ValueError:
        auc = None
    quality = dataset_quality(dataset_path)
    evaluation = {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "accuracy_std": round(float(np.std(fold_acc)), 4),
        "precision": round(float(precision_score(y, pred, pos_label="fake", zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, pos_label="fake", zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, pos_label="fake", zero_division=0)), 4),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y, pred, labels=["real", "fake"]).tolist(),
        "method": f"{k}-fold stratified cross-validation (out-of-fold predictions)",
        "dataset_rows": quality["rows"],
        "unique_articles": int(len(df)),
        "transformer_enabled": emb is not None,
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "warnings": quality["warnings"],
    }
    artifact = {**_fit_branches(X, y, emb), "transformer_enabled": emb is not None, "evaluation": evaluation}
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    m = load_metrics()
    m["training"] = evaluation
    save_metrics(m)
    _load_cached.cache_clear()
    return artifact


@lru_cache(maxsize=1)
def _load_cached(mtime: float) -> Dict:
    return joblib.load(MODEL_PATH)


def load_model() -> Dict:
    if not MODEL_PATH.exists():
        train_model()
    return _load_cached(MODEL_PATH.stat().st_mtime)


def get_training_metrics() -> Dict:
    return load_metrics().get("training") or load_model().get("evaluation", {})


# ---------------------------------------------------------------- inference
def predict_news(title: str = "", text: str = "") -> Dict:
    art = load_model()
    combined = build_combined_text(title, text)
    if not combined:
        raise ValueError("Empty input. Enter a title or article text.")

    emb = None
    if art.get("transformer_enabled") and art.get("bert_clf") is not None:
        embedder = _get_embedder()
        emb = embedder.encode([combined], show_progress_bar=False) if embedder is not None else None
    fake = float(_ensemble(art, [combined], emb)[0])
    real = 1.0 - fake

    analyzer = art["word_vec"].build_analyzer()
    unigrams = [t for t in analyzer(combined) if " " not in t]
    vocab = art["word_vec"].vocabulary_
    coverage = (sum(t in vocab for t in unigrams) / len(unigrams)) if unigrams else 0.0

    prediction = "fake" if fake >= 0.5 else "real"
    return {
        "prediction": prediction,
        "confidence": round(max(fake, real), 4),
        "probabilities": {"real": round(real, 4), "fake": round(fake, 4)},
        "vocab_coverage": round(coverage, 3),
        "top_terms": _top_terms(art, combined),
        "training_articles": art["evaluation"].get("unique_articles", 0),
    }


def _top_terms(art: Dict, combined: str, top_n: int = 8) -> List[Dict]:
    vec, clf = art["word_vec"], art["word_clf"]
    row = vec.transform([combined]).tocoo()
    names = vec.get_feature_names_out()
    sign = -1.0 if clf.classes_[1] == "real" else 1.0   # coef_ points toward classes_[1]
    terms = [{"term": names[j], "push": round(float(v * clf.coef_[0][j] * sign), 4)} for j, v in zip(row.col, row.data)]
    terms.sort(key=lambda t: abs(t["push"]), reverse=True)
    return [{**t, "toward": "fake" if t["push"] > 0 else "real"} for t in terms[:top_n]]
