import pytest

from src.explain.generate import explain_one
from src.explain.grounding import TitleIndex, check, display_title, norm_title
from src.explain.llm_client import LLMClient
from src.explain.prompt import evidence_payload, template_explanation

MOVIES = [(1, "Matrix, The (1999)", "Action|Sci-Fi|Thriller"),
          (2, "Terminator, The (1984)", "Action|Sci-Fi|Thriller"),
          (3, "Contact (1997)", "Drama|Sci-Fi"),
          (4, "Titanic (1997)", "Drama|Romance"),
          (5, "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy")]
TITLES = {m: (t, g) for m, t, g in MOVIES}
INDEX = TitleIndex.build([(m, t) for m, t, _ in MOVIES], min_tokens=1)
REC = {"movieId": 1, "title": TITLES[1][0], "genres": TITLES[1][1]}
EV = [{"movieId": 2, "title": TITLES[2][0], "genres": TITLES[2][1], "rating": 5.0},
      {"movieId": 3, "title": TITLES[3][0], "genres": TITLES[3][1], "rating": 4.5}]


def test_title_normalisation():
    assert norm_title("Matrix, The (1999)") == "the matrix"
    assert display_title("Matrix, The (1999)") == "The Matrix (1999)"
    assert norm_title("Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)") == "raiders of the lost ark"


def test_template_is_grounded_and_cites_evidence():
    text = template_explanation(evidence_payload(REC, EV))
    assert "The Matrix (1999)" in text and "The Terminator (1984)" in text and "5/5" in text
    r = check(text, REC, EV, [2, 3], INDEX)
    assert r.grounded, r.unsupported_claims


@pytest.mark.parametrize("text,needle", [
    ("We recommend The Matrix (1999) because you rated The Terminator (1984) and Contact (1997) and Titanic (1997).", "titanic"),
    ("We recommend The Matrix (1999), a 2003 hit, because you rated The Terminator (1984) and Contact (1997).", "year 2003"),
    ("We recommend The Matrix (1999), a romantic film, because you rated The Terminator (1984) and Contact (1997).", "Romance"),
    ("We recommend the Oscar-winning The Matrix (1999) because you rated The Terminator (1984) and Contact (1997).", "oscar"),
    ("We recommend The Matrix (1999) because you rated The Terminator (1984) 3/5 and Contact (1997).", "rating 3"),
    ("We recommend The Matrix (1999) because you rated The Terminator (1984).", "not named"),
    ("You will love this because you rated The Terminator (1984) and Contact (1997).", "recommended movie"),
    ("We recommend The Matrix (1999) because other users like The Terminator (1984) and Contact (1997).", "other users"),
])
def test_hallucinations_are_caught(text, needle):
    r = check(text, REC, EV, [2, 3], INDEX)
    assert not r.grounded
    assert any(needle.lower() in c.lower() for c in r.unsupported_claims), r.unsupported_claims


def test_citing_item_outside_evidence_is_flagged():
    r = check("We recommend The Matrix (1999) because you rated Toy Story (1995).", REC, EV, [5], INDEX)
    assert not r.grounded and any("not in the evidence" in c for c in r.unsupported_claims)


class FakeLLM(LLMClient):
    def __init__(self, replies):
        super().__init__(provider="ollama", model="fake")
        self.replies, self.calls = list(replies), 0

    def complete(self, system, user):
        self.calls += 1
        assert "The Terminator (1984)" in user and "Titanic" not in user   # evidence-only prompt
        return self.replies.pop(0) if self.replies else None


def test_llm_output_accepted_when_grounded():
    good = "The model picked The Matrix (1999) for you because you rated The Terminator (1984) 5/5 and Contact (1997) 4.5/5."
    llm = FakeLLM([good])
    res = explain_one(REC, EV, 2, llm, retries=1, title_index=INDEX)
    assert res["text"] == good and res["cited_items"] == [2, 3] and res["llm_name"] == "ollama:fake"
    assert res["attempts"] == 1 and not res["fell_back"]


def test_llm_retries_then_falls_back_to_template():
    bad = "We recommend The Matrix (1999), an Oscar winner, because you rated The Terminator (1984) and Contact (1997)."
    llm = FakeLLM([bad, bad])
    res = explain_one(REC, EV, 2, llm, retries=1, title_index=INDEX)
    assert llm.calls == 2 and res["llm_name"] == "template" and res["fell_back"]
    assert res["text"] == template_explanation(evidence_payload(REC, EV))


def test_llm_failure_and_template_provider():
    res = explain_one(REC, EV, 2, FakeLLM([]), retries=2, title_index=INDEX)
    assert res["llm_name"] == "template" and res["attempts"] == 1
    assert LLMClient(provider="").is_template and LLMClient(provider="template").complete("s", "u") is None
    with pytest.raises(ValueError):
        LLMClient(provider="nope")
