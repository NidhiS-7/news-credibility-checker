"""Combines independent evidence into one explainable credibility score."""
from typing import Dict, List, Optional

from explainability import analyze_language, highlight_html
from model import predict_news
from source_manager import assess_source
from verification import verify_claim

RELIABLE_AT, MISLEADING_AT = 0.68, 0.40


def _band(score: Optional[float]) -> str:
    if score is None:
        return "na"
    return "good" if score >= RELIABLE_AT else "bad" if score <= MISLEADING_AT else "mid"


def _component(key, label, score, weight, summary, detail="", short="") -> Dict:
    return {"short": short, "key": key, "label": label, "score": None if score is None else round(score, 3),
            "weight": weight if score is not None else 0.0, "band": _band(score), "summary": summary, "detail": detail}


def run_analysis(title: str, text: str, source: str, use_web: bool = True, factcheck_key: Optional[str] = None) -> Dict:
    title, text, source = (title or "").strip(), (text or "").strip(), (source or "").strip()
    lang = analyze_language(title, text)
    src = assess_source(source)
    model = predict_news(title, text)
    web = verify_claim(title, text, factcheck_key) if use_web else {"ran": False, "reliability": None, "matches": [],
                                                                      "fact_checks": [], "summary": "Web verification turned off.",
                                                                      "fact_check_verdict": None, "trusted_outlets": 0, "query": ""}
    comps: List[Dict] = []

    # ---- source
    if src.status == "trusted":
        comps.append(_component("source", "Source", 0.92, 0.25, f"{src.name} is a recognised, established outlet (matched by {src.matched_by}).",
                                "The source is self-reported. Paste the article's URL instead of a name so it can be verified by domain.", f"{src.name}, established outlet"))
    elif src.status == "unreliable":
        comps.append(_component("source", "Source", 0.08, 0.30, f"{src.name} is on the list of known unreliable sources."))
    elif src.status == "unknown":
        comps.append(_component("source", "Source", 0.45, 0.12, f"'{src.name}' is on neither list. That says little either way; look at who runs it and whether others cite it.", "", "Unknown source"))
    else:
        comps.append(_component("source", "Source", None, 0, "No source given. Adding the outlet name or article URL improves the check.", "", "Not provided"))

    # ---- language
    risk = [f for f in lang["findings"] if f["kind"] == "risk"]
    if risk:
        lsum = "Sensational or manipulative wording: " + "; ".join(f["title"].lower() for f in risk[:3]) + "."
    elif any(f["kind"] == "positive" for f in lang["findings"]):
        lsum = "Neutral, attributed reporting style. Style is only a weak indicator of truth."
    else:
        lsum = "No red-flag wording found. A calm tone alone doesn't make a claim true."
    comps.append(_component("language", "Writing style", lang["score"], 0.20, lsum, "", "Sensational wording" if risk else "No red flags"))

    # ---- classifier (weight scales with training size and how familiar the text looks to it)
    size_w = max(0.04, min(0.30, model["training_articles"] / 5000))
    weight = round(size_w * max(0.25, min(1.0, model["vocab_coverage"] * 1.6)), 3)
    msum = (f"Text classifier leans {model['prediction'].upper()} ({model['confidence']:.0%}). "
            f"It learned from only {model['training_articles']} articles, so it counts for {weight:.0%} of the score.")
    comps.append(_component("model", "Text classifier", model["probabilities"]["real"], weight, msum,
                            f"The classifier recognised {model['vocab_coverage']:.0%} of the words in this text.", f"Leans {model['prediction']} ({model['confidence']:.0%})"))

    # ---- web
    if web["ran"] or web.get("fact_check_verdict"):
        w = 0.40 if (web["matches"] or web.get("fact_check_verdict")) else 0.25
        comps.append(_component("web", "Web cross-check", web["reliability"], w, web["summary"], "",
                                "Disputed by fact-checks" if web.get("debunked") else f"{web['trusted_outlets']} trusted outlet(s)" if web["trusted_outlets"] else "No trusted coverage"))
    else:
        comps.append(_component("web", "Web cross-check", None, 0, web["summary"], "", "Not checked"))

    total = sum(c["weight"] for c in comps)
    score = sum(c["weight"] * c["score"] for c in comps if c["score"] is not None) / total if total else 0.5

    fc = web.get("fact_check_verdict")
    if fc == "false":
        score = min(score, 0.25)
    elif fc == "true":
        score = max(score, 0.75)
    debunked = bool(web.get("debunked"))
    if debunked:
        score = min(score, 0.25)
    if src.status == "unreliable":
        score = min(score, 0.45)

    corroborated = bool(web["ran"] and web["trusted_outlets"]) or fc == "true" or (src.status == "trusted" and src.matched_by == "domain")
    capped = score >= RELIABLE_AT and not corroborated
    if capped:  # a typed outlet name can be faked; demand independent confirmation before "reliable"
        score = RELIABLE_AT - 0.01

    verdict = "Likely reliable" if score >= RELIABLE_AT else "Likely misleading" if score <= MISLEADING_AT else "Needs verification"
    external = int(src.status in ("trusted", "unreliable")) + int(bool(web["ran"] and (web["matches"] or fc)))
    strength = ["Low", "Medium", "High"][external]

    summary = {
        "Likely reliable": "Several independent signals point the same way. Still confirm anything you plan to act on or share.",
        "Needs verification": "The evidence is mixed or thin. Treat this as unconfirmed until a trusted outlet or fact-checker reports it.",
        "Likely misleading": "Multiple warning signs. Don't share this before checking it against a trusted source.",
    }[verdict]
    if capped:
        summary = "The style and source look fine, but nothing independent confirms the story yet. Add the article URL or enable web checks."
    if debunked and fc != "false":
        summary = "Coverage of this claim is debunking it rather than confirming it."
    if fc == "false":
        summary = "A fact-checking organisation has rated a matching claim as false."

    steps = ["Search the headline on a trusted outlet's own site and compare the details.",
             "Check the publication date; old stories are often recirculated as new."]
    if not source:
        steps.append("Find out where the story originally appeared and who is quoted.")
    if lang["spans"]:
        steps.append("Be wary of the highlighted phrases: genuine reporting rarely relies on them.")
    if web.get("matches"):
        steps.append("Open the matching coverage listed under Web matches and compare numbers, names and quotes.")

    return {"verdict": verdict, "score": int(round(score * 100)), "strength": strength, "summary": summary,
            "components": comps, "language": {k: v for k, v in lang.items() if k not in ("spans",)},
            "highlighted_html": highlight_html(lang["display_text"], lang["spans"]),
            "source": src.__dict__, "model": model, "web": web, "next_steps": steps}
