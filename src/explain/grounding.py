"""Rule-based grounding and hallucination checks for explanations (M4).

An explanation is GROUNDED when
  1. it names the recommended movie and every item it cites, and every cited item
     is in the evidence it was given;
  2. it mentions no other movie titles;
  3. every year, genre and rating it states is backed by the evidence;
  4. it gives no reason outside the evidence (awards, popularity, cast, "other
     users", ...).
Every violation is reported as a short string in `unsupported_claims`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama",
          "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance", "Sci-Fi",
          "Thriller", "War", "Western"]

# surface forms -> MovieLens genre
GENRE_PATTERNS = {
    "Action": r"\baction\b", "Adventure": r"\badventures?\b", "Animation": r"\banimat(?:ion|ed)\b",
    "Children": r"\b(?:children'?s?|kids?|family films?)\b", "Comedy": r"\b(?:comed(?:y|ies)|comedic)\b",
    "Crime": r"\bcrime\b", "Documentary": r"\bdocumentar(?:y|ies)\b", "Drama": r"\bdrama(?:s|tic)?\b",
    "Fantasy": r"\bfantasy\b", "Film-Noir": r"\b(?:film[- ]noir|noir)\b", "Horror": r"\bhorror\b",
    "IMAX": r"\bimax\b", "Musical": r"\bmusicals?\b", "Mystery": r"\bmyster(?:y|ies)\b",
    "Romance": r"\b(?:romance|romantic|rom-?coms?)\b", "Sci-Fi": r"\b(?:sci[- ]?fi|science[- ]fiction)\b",
    "Thriller": r"\bthrillers?\b", "War": r"\bwar\b", "Western": r"\bwesterns?\b",
}

# reasons the evidence can never support
UNSUPPORTED_REASONS = [
    r"\bawards?\b", r"\boscars?\b", r"\bacclaimed\b", r"\bcritic(?:s|al|ally)?\b", r"\bpopular(?:ity)?\b",
    r"\btrending\b", r"\bblockbuster\b", r"\bbox[- ]office\b", r"\bdirect(?:or|ed)\b", r"\bstarring\b",
    r"\bact(?:or|ors|ress|resses)\b", r"\bcast\b", r"\beveryone\b", r"\bmost (?:people|viewers)\b",
    r"\bother (?:users|viewers|people)\b", r"\busers like you\b", r"\bfans of\b", r"\bmasterpiece\b",
    r"\biconic\b", r"\bcult classic\b", r"\bhighly rated\b", r"\bbest[- ]selling\b", r"\bsequel\b",
    r"\bsame (?:director|studio|franchise|universe)\b", r"\bbased on\b", r"\btrue story\b",
]

RATING_RE = re.compile(r"(\d(?:\.\d)?)\s*(?:/\s*5|out of 5|stars?|★)", re.I)
YEAR_RE = re.compile(r"\b(18[89]\d|19\d\d|20\d\d)\b")


def norm_title(title: str) -> str:
    """'Matrix, The (1999)' -> 'the matrix'; lowercase, punctuation stripped."""
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", title or "").strip()
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t)     # trailing alternate title
    m = re.match(r"^(.*), (The|A|An|Les|La|Le|Il|El|Das|Die|Der)$", t)
    if m:
        t = f"{m.group(2)} {m.group(1)}"
    t = re.sub(r"\s*\([^)]*\)", "", t)            # drop alternate titles in brackets
    return norm_text(t)


def norm_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split())


def display_title(title: str) -> str:
    """'Matrix, The (1999)' -> 'The Matrix (1999)' for human-facing text."""
    m = re.match(r"^(.*?), (The|A|An|Les|La|Le|Il|El|Das|Die|Der)((?: \([^()]*\))*)$", title or "")
    return f"{m.group(2)} {m.group(1)}{m.group(3)}" if m else (title or "")


@dataclass
class TitleIndex:
    """n-gram index of catalog titles for spotting movie mentions in free text."""

    by_norm: dict = field(default_factory=dict)   # normalized title -> set(movieId)
    max_len: int = 1
    min_len: int = 2

    @classmethod
    def build(cls, movies: list[tuple[int, str]], min_tokens: int = 2) -> "TitleIndex":
        idx = cls(min_len=min_tokens)
        for mid, title in movies:
            n = norm_title(title)
            if len(n.split()) >= min_tokens:   # single-word titles ("Up", "Heat") are too ambiguous
                idx.by_norm.setdefault(n, set()).add(int(mid))
                idx.max_len = max(idx.max_len, len(n.split()))
        return idx

    def mentions(self, text: str) -> dict[str, set]:
        toks = norm_text(text).split()
        found = {}
        for n in range(self.min_len, min(self.max_len, len(toks)) + 1):
            for i in range(len(toks) - n + 1):
                g = " ".join(toks[i:i + n])
                if g in self.by_norm:
                    found[g] = self.by_norm[g]
        return found


@dataclass
class GroundingResult:
    grounded: bool
    unsupported_claims: list[str]
    cited_present: bool
    rec_present: bool
    n_words: int
    flesch: float


def _syllables(word: str) -> int:
    groups = re.findall(r"[aeiouy]+", word.lower())
    n = len(groups) - (1 if word.lower().endswith("e") and len(groups) > 1 else 0)
    return max(1, n)


def flesch_reading_ease(text: str) -> float:
    words = re.findall(r"[A-Za-z']+", text)
    sents = max(1, len(re.findall(r"[.!?]+", text)))
    if not words:
        return 0.0
    syl = sum(_syllables(w) for w in words)
    return 206.835 - 1.015 * len(words) / sents - 84.6 * syl / len(words)


def check(text: str, rec: dict, evidence: list[dict], cited_ids: list[int],
          title_index: TitleIndex | None = None) -> GroundingResult:
    """rec: {movieId, title, genres}; evidence: [{movieId, title, genres, rating}] given to the writer."""
    claims: list[str] = []
    nt = norm_text(text)
    ev_by_id = {int(e["movieId"]): e for e in evidence}

    # 1. cited items exist in evidence and are named; recommended item is named
    for c in cited_ids:
        if int(c) not in ev_by_id:
            claims.append(f"cites item {c} that is not in the evidence")
    cited_present = bool(cited_ids) and all(
        int(c) in ev_by_id and norm_title(ev_by_id[int(c)]["title"]) in nt for c in cited_ids)
    if not cited_ids:
        claims.append("cites no evidence")
    elif not cited_present:
        claims.append("a cited evidence item is not named in the text")
    rec_present = norm_title(rec["title"]) in nt
    if not rec_present:
        claims.append("recommended movie is not named")

    allowed = [rec] + list(evidence)
    allowed_norm = {norm_title(a["title"]) for a in allowed}
    allowed_ids = {int(a["movieId"]) for a in allowed}
    # remove every surface form of the allowed titles, so their words can neither be
    # read as other titles/genres nor join neighbouring words into a false title
    masked = nt
    forms = set()
    for a in allowed:
        forms |= {norm_text(a["title"]), norm_text(display_title(a["title"])), norm_title(a["title"])}
    for f in sorted(forms, key=len, reverse=True):
        if f:
            masked = re.sub(r"(?<![a-z0-9])" + re.escape(f) + r"(?![a-z0-9])", " xsep ", masked)

    # 2. other movie titles
    if title_index is not None:
        for g, ids in title_index.mentions(masked).items():
            if g in allowed_norm or ids & allowed_ids:
                continue
            if any(g in a for a in allowed_norm):    # sub-phrase of an allowed title
                continue
            claims.append(f"mentions movie not in evidence: '{g}'")

    # 3a. years
    allowed_years = set()
    for a in allowed:
        allowed_years.update(YEAR_RE.findall(a["title"] or ""))
    for y in set(YEAR_RE.findall(text)):
        if y not in allowed_years:
            claims.append(f"year {y} not in evidence")
    # 3b. genres
    allowed_genres = set()
    for a in allowed:
        allowed_genres.update((a.get("genres") or "").split("|"))
    text_wo_titles = masked
    for g, pat in GENRE_PATTERNS.items():
        if re.search(pat, text_wo_titles, re.I) and g not in allowed_genres:
            claims.append(f"genre '{g}' not in evidence")
    # 3c. ratings
    allowed_ratings = {float(e["rating"]) for e in evidence if e.get("rating") is not None}
    for v in RATING_RE.findall(text):
        if float(v) not in allowed_ratings:
            claims.append(f"rating {v} not in evidence")

    # 4. reasons outside the evidence
    for pat in UNSUPPORTED_REASONS:
        m = re.search(pat, text_wo_titles, re.I)
        if m:
            claims.append(f"unsupported reason: '{m.group(0)}'")

    return GroundingResult(not claims, claims, cited_present, rec_present,
                           len(re.findall(r"\S+", text)), round(flesch_reading_ease(text), 1))
