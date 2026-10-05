"""Language-level red flags. This measures *style*, not truth: a calm tone never proves a claim."""
import html
import re
from typing import Dict, List, Tuple

SIGNAL_GROUPS = [
    {"key": "miracle", "label": "Miracle or cure claims", "weight": 0.22, "cap": 0.50, "patterns": [
        r"\bmiracles?\b", r"\bcures?\s+(?:all|any|every|cancer|diabetes|aids|covid)\w*", r"\binstantly\b",
        r"\bregrow\w*", r"\ball diseases\b", r"\bone (?:leaf|herb|pill|drink|fruit)\b", r"\bdetox\w*",
        r"\bdoctors hate\b", r"\bovernight (?:cure|relief|results?)\b", r"\bancient (?:secret|remedy|cure|herb)s?\b"]},
    {"key": "secrecy", "label": "Secrecy and conspiracy framing", "weight": 0.20, "cap": 0.45, "patterns": [
        r"\bsecrets?\b", r"\bhidden (?:truth|cure|agenda)\b", r"don'?t want you to know", r"\bcover[- ]?ups?\b",
        r"\bleaked\b", r"\bexposed\b", r"\bsuppressed\b", r"\bbig pharma\b", r"\bwhat (?:they|the media) (?:won'?t|aren'?t) (?:tell|telling)"]},
    {"key": "clickbait", "label": "Clickbait phrasing", "weight": 0.18, "cap": 0.40, "patterns": [
        r"you won'?t believe", r"\bshocking\b", r"\bgoes? viral\b", r"\bviral\b", r"mind[- ]?blowing",
        r"what happens next", r"\b\d+ (?:reasons|things|secrets|tricks)\b", r"\bmust[- ]see\b", r"\bstunned\b"]},
    {"key": "urgency", "label": "Pressure to share", "weight": 0.25, "cap": 0.50, "patterns": [
        r"share (?:this|now|before|with everyone)", r"forward this", r"before it'?s deleted", r"\bwake up\b", r"\burgent(?:ly)?\b"]},
    {"key": "extraordinary", "label": "Extraordinary claims", "weight": 0.30, "cap": 0.60, "patterns": [
        r"\bcure[sd]?\b.{0,40}\b(?:cancer|diabetes|hiv|aids|covid)\b", r"\b(?:every|all) (?:citizens?|indians?|people|households?|families)\b.{0,30}\b(?:get|receive)\b",
        r"\bfree\s+(?:rs\.?\s?|\u20b9\s?|\$\s?)?\d[\d,]*", r"\bban(?:s|ned)? (?:all|every)\b", r"\bfrom tomorrow\b",
        r"\b(?:live|living) forever\b", r"\bwhatsapp\b.{0,40}\b(?:forward|message|rumou?r)\b", r"\bnasa (?:confirms?|warns?)\b.{0,40}\b(?:darkness|apocalypse|end of the world)\b"]},
    {"key": "absolutes", "label": "Absolute or unverifiable certainty", "weight": 0.15, "cap": 0.30, "patterns": [
        r"\bguaranteed\b", r"\b100\s?%", r"\bproven fact\b", r"\bundeniable\b", r"\bno one (?:knows|tells)\b"]},
]
ACRONYMS = {"NASA", "ISRO", "COVID", "NATO", "UNESCO", "ASEAN", "OPEC", "FIFA", "INDIA", "CNN", "NDTV"}
ATTRIBUTION = re.compile(
    r"\b(said|says|according to|told reporters|stated|announced|confirmed|spokesperson|officials?|"
    r"ministry|press release|published in|researchers?|statement)\b", re.I)


def _spans(patterns: List[str], text: str) -> List[Tuple[int, int, str]]:
    found = []
    for p in patterns:
        for m in re.finditer(p, text, re.I):
            found.append((m.start(), m.end(), m.group(0).lower()))
    return found


def analyze_language(title: str, text: str) -> Dict:
    display = f"{title.strip()}\n\n{text.strip()}".strip()
    words = re.findall(r"\b\w+\b", display)
    findings, all_spans, risk = [], [], 0.0

    for g in SIGNAL_GROUPS:
        hits = _spans(g["patterns"], display)
        if not hits:
            continue
        phrases = sorted({h[2] for h in hits})
        weight = min(g["cap"], g["weight"] * len(phrases))
        risk += weight
        all_spans += [(s, e) for s, e, _ in hits]
        findings.append({"kind": "risk", "title": g["label"], "phrases": phrases, "weight": round(weight, 2),
                         "detail": f"Found: {', '.join(phrases)}."})

    caps = [w for w in re.findall(r"\b[A-Z]{3,}\b", display) if w not in ACRONYMS]
    if len(caps) >= 2 and words and len(caps) / len(words) >= 0.06:
        risk += 0.15
        findings.append({"kind": "risk", "title": "Heavy use of capital letters", "phrases": caps[:5], "weight": 0.15,
                         "detail": f"{len(caps)} all-caps words (e.g. {', '.join(caps[:3])})."})
    bangs = display.count("!")
    if bangs >= 2:
        w = min(0.2, 0.06 * bangs)
        risk += w
        findings.append({"kind": "risk", "title": "Excessive exclamation marks", "phrases": [], "weight": round(w, 2),
                         "detail": f"{bangs} exclamation marks; straight news reporting rarely uses them."})

    body_words = len(re.findall(r"\b\w+\b", text))
    attributions = {m.group(0).lower() for m in ATTRIBUTION.finditer(text)}
    bonus = 0.0
    if body_words >= 40:
        if len(attributions) >= 2:
            bonus = 0.12
            findings.append({"kind": "positive", "title": "Attributes claims to people or institutions", "phrases": sorted(attributions)[:4], "weight": 0.12,
                             "detail": "The article points to who said or confirmed things, which makes claims checkable."})
        elif not attributions:
            risk += 0.15
            findings.append({"kind": "risk", "title": "No attribution", "phrases": [], "weight": 0.15,
                             "detail": "A full-length article that never says who said or confirmed anything."})
    elif body_words == 0:
        findings.append({"kind": "info", "title": "Headline only", "phrases": [], "weight": 0,
                         "detail": "Only a headline was provided, so style analysis is limited. Paste the article body for a better check."})

    risk = min(1.0, risk)
    # A neutral tone is only weak evidence of reliability, so the ceiling is deliberately modest.
    score = max(0.05, min(0.70, 0.50 - 0.50 * risk + bonus))
    return {"score": round(score, 3), "risk": round(risk, 3), "findings": findings,
            "display_text": display, "spans": all_spans}


def highlight_html(display_text: str, spans: List[Tuple[int, int]]) -> str:
    merged: List[List[int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, pos = [], 0
    for s, e in merged:
        out.append(html.escape(display_text[pos:s]))
        out.append(f'<mark class="flag">{html.escape(display_text[s:e])}</mark>')
        pos = e
    out.append(html.escape(display_text[pos:]))
    return "".join(out).replace("\n", "<br>")
