"""Live cross-checking: (1) do trusted outlets report the same story? (2) has a fact-checker rated it?"""
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional

import requests

from source_manager import assess_source
from utils import first_sentence

NEWS_RSS = "https://news.google.com/rss/search"
FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
HEADERS = {"User-Agent": "Mozilla/5.0 (NewsCredibilityChecker/2.0)"}
TIMEOUT = 8
_STOP = set("the and for with that this from have has had was were are will would been about into over after "
            "before their they them its his her than then also but not you your our who what when where which".split())
FALSE_WORDS = ("not true", "mostly false", "false", "fake", "incorrect", "misleading", "pants on fire", "fabricated",
               "hoax", "untrue", "no evidence", "satire", "scam", "wrong", "manipulated")
DEBUNK_WORDS = ("fact check", "fact-check", "fact checked", "debunk", "hoax", "myth", "fake", "misleading", "no evidence",
                "not true", "rumour", "rumor", "baseless", "unfounded", "falsely", "false claim", "scam", "satire",
                "does not cure", "doesn't cure", "won't cure", "viral claim", "clarifies", "denies")
MIXED_WORDS = ("mostly true", "mixture", "partly", "half", "unproven", "missing context", "outdated", "exaggerat", "unverified")
TRUE_WORDS = ("true", "correct", "accurate", "legit", "real")


@dataclass
class WebEvidence:
    ran: bool = False
    error: Optional[str] = None
    query: str = ""
    matches: List[Dict] = field(default_factory=list)
    trusted_outlets: int = 0
    debunked: bool = False
    debunk_count: int = 0
    fact_checks: List[Dict] = field(default_factory=list)
    fact_check_verdict: Optional[str] = None   # false | mixed | true | None
    fact_check_note: Optional[str] = None
    reliability: Optional[float] = None
    summary: str = ""


def _tokens(s: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower()) if len(t) > 2 and t not in _STOP}


def _containment(q: set, t: set) -> float:
    return len(q & t) / len(q) if q else 0.0


def build_query(title: str, text: str) -> str:
    base = (title or "").strip() or first_sentence(text)
    base = re.sub(r"[^\w\s'-]", " ", base)
    return " ".join(base.split()[:14])


def search_news(query: str, limit: int = 20) -> List[Dict]:
    r = requests.get(NEWS_RSS, params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    items = []
    for it in ET.fromstring(r.content).iter("item"):
        title = it.findtext("title", "")
        src = it.find("source")
        name = (src.text or "") if src is not None else ""
        if name and title.endswith(f" - {name}"):
            title = title[: -(len(name) + 3)]
        try:
            published = parsedate_to_datetime(it.findtext("pubDate", "")).strftime("%d %b %Y")
        except Exception:
            published = ""
        items.append({"title": title, "source": name, "source_url": src.get("url", "") if src is not None else "",
                      "url": it.findtext("link", ""), "published": published})
        if len(items) >= limit:
            break
    return items


def _classify_rating(rating: str) -> str:
    r = (rating or "").lower()
    if any(w in r for w in FALSE_WORDS):
        return "false"
    if any(w in r for w in MIXED_WORDS):
        return "mixed"
    if any(w in r for w in TRUE_WORDS):
        return "true"
    return "mixed"


def search_fact_checks(query: str, api_key: str) -> List[Dict]:
    r = requests.get(FACTCHECK_URL, params={"query": query, "key": api_key, "languageCode": "en", "pageSize": 10}, timeout=TIMEOUT)
    r.raise_for_status()
    out = []
    for c in r.json().get("claims", []):
        review = (c.get("claimReview") or [{}])[0]
        out.append({"claim": c.get("text", ""), "claimant": c.get("claimant", ""),
                    "publisher": (review.get("publisher") or {}).get("name", ""), "url": review.get("url", ""),
                    "rating": review.get("textualRating", ""), "class": _classify_rating(review.get("textualRating", ""))})
    return out


def verify_claim(title: str, text: str, factcheck_key: Optional[str] = None) -> Dict:
    ev = WebEvidence(query=build_query(title, text))
    qt = _tokens(ev.query)
    if len(qt) < 3:
        ev.error = "Not enough distinctive words to search for. Add a fuller headline."
        return asdict(ev)

    try:
        for item in search_news(ev.query):
            sim = _containment(qt, _tokens(item["title"]))
            if sim >= 0.5:
                item["similarity"] = round(sim, 2)
                item["trusted"] = assess_source(item["source_url"] or item["source"]).status == "trusted"
                item["debunk"] = any(w in item["title"].lower() for w in DEBUNK_WORDS)
                ev.matches.append(item)
        ev.ran = True
    except Exception as exc:
        ev.error = f"News search unavailable ({type(exc).__name__}). Check your internet connection."
    ev.matches.sort(key=lambda m: (not m["trusted"], -m["similarity"]))
    ev.trusted_outlets = len({m["source"] for m in ev.matches if m["trusted"] and not m["debunk"]})
    ev.debunk_count = sum(m["debunk"] for m in ev.matches)
    ev.debunked = any(m["debunk"] and m["trusted"] for m in ev.matches) or ev.debunk_count >= 2

    key = factcheck_key or os.getenv("GOOGLE_FACTCHECK_API_KEY")
    if key:
        try:
            checks = [c for c in search_fact_checks(ev.query, key) if _containment(qt, _tokens(c["claim"])) >= 0.4]
            ev.fact_checks = checks
            if checks:
                votes = [c["class"] for c in checks]
                ev.fact_check_verdict = "false" if "false" in votes else ("true" if votes.count("true") > votes.count("mixed") else "mixed")
        except Exception as exc:
            ev.fact_check_note = f"Fact-check lookup failed ({type(exc).__name__})."
    else:
        ev.fact_check_note = "Fact-check database not connected (add a Google Fact Check API key to enable)."

    news_score = None
    if ev.ran:
        n, other = ev.trusted_outlets, len([m for m in ev.matches if not m["debunk"]])
        news_score = 0.08 if ev.debunked else 0.92 if n >= 3 else 0.82 if n == 2 else 0.70 if n == 1 else 0.48 if other >= 2 else 0.40 if other == 1 else 0.28
    fc_score = {"false": 0.05, "mixed": 0.5, "true": 0.95}.get(ev.fact_check_verdict)
    if fc_score is not None:
        ev.reliability = fc_score if news_score is None else 0.75 * fc_score + 0.25 * news_score
    else:
        ev.reliability = news_score

    if ev.debunked and not ev.fact_check_verdict:
        ev.summary = "News coverage of this claim is fact-checking or disputing it, not confirming it."
    elif ev.fact_check_verdict:
        ev.summary = f"A fact-checking organisation has rated a matching claim as {ev.fact_check_verdict}."
    elif not ev.ran:
        ev.summary = ev.error or "Web check did not run."
    elif ev.trusted_outlets:
        ev.summary = f"{ev.trusted_outlets} trusted outlet(s) are reporting a matching story."
    elif ev.matches:
        ev.summary = "Only lesser-known sites carry a matching story; no trusted outlet was found."
    else:
        ev.summary = "No matching coverage found. That is a weak warning sign, not proof: very fresh or very local stories often have none yet."
    return asdict(ev)
