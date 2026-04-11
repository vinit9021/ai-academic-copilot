import json
import os
from collections import defaultdict


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROGRESS_FILE = os.path.join(PROJECT_ROOT, "progress.json")


def _normalize_topic(topic: str) -> str:
    return str(topic).strip().lower()


def _empty_data():
    return {"topics": defaultdict(list)}


def load_data():
    data = _empty_data()

    if not os.path.exists(PROGRESS_FILE):
        return data

    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as file_handle:
            raw_data = json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        return data

    topics = raw_data.get("topics", {}) if isinstance(raw_data, dict) else {}
    for topic, scores in topics.items():
        normalized_topic = _normalize_topic(topic)
        if not normalized_topic:
            continue

        if isinstance(scores, list):
            cleaned_scores = []
            for score in scores:
                try:
                    cleaned_scores.append(int(score))
                except (TypeError, ValueError):
                    continue
            data["topics"][normalized_topic] = cleaned_scores

    return data


def save_data(data):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)

    topics = data.get("topics", defaultdict(list))
    serializable = {"topics": {topic: scores for topic, scores in topics.items()}}

    with open(PROGRESS_FILE, "w", encoding="utf-8") as file_handle:
        json.dump(serializable, file_handle, indent=2, ensure_ascii=False)


def update_weakness(topic: str, score: int):
    data = load_data()
    normalized_topic = _normalize_topic(topic)

    if not normalized_topic:
        return

    try:
        cleaned_score = int(score)
    except (TypeError, ValueError):
        cleaned_score = 0

    data["topics"][normalized_topic].append(max(0, min(10, cleaned_score)))
    save_data(data)


def get_weak_topics(threshold=5):
    data = load_data()
    weak_topics = {}

    for topic, scores in data["topics"].items():
        if not scores:
            continue

        average_score = sum(scores) / len(scores)
        if average_score < threshold:
            weak_topics[topic] = round(average_score, 2)

    return weak_topics


def generate_report():
    weak_topics = get_weak_topics()

    if not weak_topics:
        return "✅ No weak topics found yet. Keep going strong!"

    lines = ["⚠️ Weak Areas (All Sessions):", ""]

    for topic, average_score in sorted(weak_topics.items(), key=lambda item: item[1]):
        lines.append(f"* {topic} → avg score: {average_score}/10")

    lines.append("")
    lines.append("👉 Focus more on these topics.")

    return "\n".join(lines)