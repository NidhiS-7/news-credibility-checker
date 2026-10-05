import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from typing import List

from storage import DATA_DIR
from utils import extract_domain

TRUSTED_SOURCE_PATH = DATA_DIR / "trusted_sources.json"


@dataclass
class SourceAssessment:
    status: str          # trusted | unreliable | unknown | none
    name: str = ""
    matched_by: str = ""  # domain | name | similar name


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = "".join(ch for ch in s if ch.isalnum() or ch == " ")
    s = " ".join(s.split())
    return s[4:] if s.startswith("the ") else s


@lru_cache(maxsize=4)
def _load(mtime: float):
    with open(TRUSTED_SOURCE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("trusted", []), data.get("unreliable", [])


def _entries():
    try:
        return _load(TRUSTED_SOURCE_PATH.stat().st_mtime)
    except (OSError, json.JSONDecodeError):
        return [], []


def trusted_domains() -> List[str]:
    return [d for e in _entries()[0] for d in e.get("domains", [])]


def _match(entries, raw: str, domain: str):
    norm = _norm(raw)
    for e in entries:
        for d in e.get("domains", []):
            if domain and (domain == d or domain.endswith("." + d)):
                return e["name"], "domain"
    for e in entries:
        names = {_norm(e["name"])} | {_norm(a) for a in e.get("aliases", [])}
        if norm in names:
            return e["name"], "name"
    if len(norm) > 4:  # typo tolerance, only for longer names ("reuter", "hindustan time")
        for e in entries:
            names = {_norm(e["name"])} | {_norm(a) for a in e.get("aliases", [])}
            if any(len(n) > 4 and SequenceMatcher(None, norm, n).ratio() >= 0.9 for n in names):
                return e["name"], "similar name"
    return None


def assess_source(source: str) -> SourceAssessment:
    if not source or not source.strip():
        return SourceAssessment("none")
    raw, domain = source.strip(), extract_domain(source)
    trusted, unreliable = _entries()
    # unreliable list is checked first so a bad outlet can never be waved through
    hit = _match(unreliable, raw, domain)
    if hit:
        return SourceAssessment("unreliable", *hit)
    hit = _match(trusted, raw, domain)
    if hit:
        return SourceAssessment("trusted", *hit)
    return SourceAssessment("unknown", domain or raw)


def is_source_trusted(source_name: str) -> bool:
    return assess_source(source_name).status == "trusted"
