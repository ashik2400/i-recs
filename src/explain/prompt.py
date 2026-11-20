"""Evidence-only prompt and the deterministic template explanation."""
from __future__ import annotations

import json

from src.explain.grounding import display_title

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You write one short explanation for a movie recommendation.
You are given structured EVIDENCE computed from the recommender model. Rules:
- Use ONLY facts in the evidence. Do not add any other reason, movie, person, year, genre, award, popularity or plot detail.
- Name the recommended movie and each evidence movie exactly as written in the evidence (title with year).
- You may mention the user's rating of an evidence movie and, for that movie only, the genres it shares with the recommended movie. Nothing else.
- Say that these past ratings are what led the model to this recommendation.
- 1-2 sentences, at most 50 words, plain text, no lists, no quotes around titles."""


def _genres(g: str | None) -> set:
    return set((g or "").split("|")) - {"", "(no genres listed)"}


def evidence_payload(rec: dict, cited: list[dict]) -> dict:
    """Structured facts the writer may use. rec/cited carry title, genres (and rating for cited)."""
    rg = _genres(rec.get("genres"))
    return {
        "recommended_movie": display_title(rec["title"]),
        "evidence": [{"movie": display_title(c["title"]), "your_rating": float(c["rating"]),
                      "genres_shared_with_recommended": sorted(rg & _genres(c.get("genres")))} for c in cited],
        "why_these": "Removing these ratings from your history lowers the model's score for the recommended movie the most.",
    }


def user_prompt(payload: dict) -> str:
    return "EVIDENCE:\n" + json.dumps(payload, indent=2) + "\n\nWrite the explanation."


def _fmt_rating(r: float) -> str:
    return f"{r:g}/5"


def template_explanation(payload: dict) -> str:
    ev = payload["evidence"]
    rec = payload["recommended_movie"]
    if not ev:
        return f"We recommend {rec}."
    parts = [f"{e['movie']} ({_fmt_rating(e['your_rating'])})" for e in ev]
    liked = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    text = f"We recommend {rec} because you rated {liked}; these ratings did the most to put it in your list."
    best = max(ev, key=lambda e: len(e["genres_shared_with_recommended"]))
    if best["genres_shared_with_recommended"]:
        g = best["genres_shared_with_recommended"]
        g_txt = g[0] if len(g) == 1 else ", ".join(g[:-1]) + " and " + g[-1]
        text += f" Like {best['movie']}, it is {g_txt}."
    return text
