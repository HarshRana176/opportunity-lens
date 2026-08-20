"""
Shared fixtures for the characterization test suite.

Importing `app.extractor` (or any of app.pdf/app.llm/app.experience/
app.skills) is safe without Ollama running: the ChatOllama client and
LangChain chains are constructed lazily in app.llm and only make network
calls when `.invoke(...)` is called. No test in this suite calls the real
`extraction_chain` or `skill_classifier_chain` — both are monkeypatched.
"""
import datetime as dt

import pytest

# A deliberately fixed, obviously-not-real "today" so date-dependent
# behavior (the "Present" -> date.today() path) is exercised
# deterministically instead of depending on the day the suite runs.
FROZEN_TODAY = dt.date(2030, 1, 15)


class _FrozenDate(dt.date):
    """Stand-in for `datetime.date` whose `.today()` is pinned."""

    @classmethod
    def today(cls):
        return FROZEN_TODAY


@pytest.fixture
def frozen_today(monkeypatch):
    """Pin `app.experience.date.today()` to FROZEN_TODAY for this test.

    parse_resume_date() lives in app.experience and reads `date` as a
    name in ITS OWN module globals, so the patch target must be
    app.experience (not app.extractor, even though app.extractor
    re-exports parse_resume_date) -- Python resolves a function's
    globals from the module it was defined in, not the module it was
    imported through.
    """
    import app.experience as experience

    monkeypatch.setattr(experience, "date", _FrozenDate)
    return FROZEN_TODAY
