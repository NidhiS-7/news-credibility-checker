import re
from typing import Optional
from urllib.parse import urlparse

_URL_RE = re.compile(r"^(?:https?://)?(?:[\w-]+\.)+[a-z]{2,}(?:[/:?#].*)?$", re.I)


def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_combined_text(title: str = "", text: str = "") -> str:
    """Text fed to the classifier. The source name is deliberately NOT included:
    source credibility is judged separately, otherwise the model just memorises outlet names."""
    title, text = clean_text(title), clean_text(text)
    return f"{title}. {text}".strip(" .") if (title and text) else (title or text)


def extract_domain(value: Optional[str]) -> str:
    """Return the bare domain if `value` looks like a URL / domain, else ''."""
    v = (value or "").strip().lower()
    if not v or " " in v or not _URL_RE.match(v):
        return ""
    if "://" not in v:
        v = "http://" + v
    host = urlparse(v).hostname or ""
    return host[4:] if host.startswith("www.") else host


def first_sentence(text: str, max_words: int = 18) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return " ".join(sentence.split()[:max_words])
