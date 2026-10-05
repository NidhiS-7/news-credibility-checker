import json
import os
import time
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
METRICS_PATH = BASE_DIR / "metrics.json"
HISTORY_PATH = DATA_DIR / "history.jsonl"


def load_metrics() -> Dict:
    if not METRICS_PATH.exists():
        return {"articles_analyzed": 0, "training": {}}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"articles_analyzed": 0, "training": {}}


def save_metrics(metrics: Dict) -> None:
    tmp = METRICS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    os.replace(tmp, METRICS_PATH)


def log_analysis(result: Dict, title: str, source: str) -> None:
    """Store a small, privacy-friendly record (no article body) and bump the counter."""
    metrics = load_metrics()
    metrics["articles_analyzed"] = int(metrics.get("articles_analyzed", 0)) + 1
    save_metrics(metrics)
    HISTORY_PATH.parent.mkdir(exist_ok=True)
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "title": (title or "")[:120],
        "source": (source or "")[:80],
        "verdict": result["verdict"],
        "score": result["score"],
        "web_checked": result["web"].get("ran", False),
    }
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_history(limit: int = 500) -> List[Dict]:
    if not HISTORY_PATH.exists():
        return []
    lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()[-limit:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
