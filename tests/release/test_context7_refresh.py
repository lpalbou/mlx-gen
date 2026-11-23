import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "context7_refresh", Path(__file__).resolve().parents[2] / "tools" / "context7_refresh.py"
)
context7_refresh = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(context7_refresh)
Context7Refresh = context7_refresh.Context7Refresh

LIBRARY = "/lpalbou/mlx-gen"
URL = "https://github.com/lpalbou/mlx-gen"


class _Api:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, path: str, body: dict):
        self.calls.append((path, body))
        return self.responses.pop(0)


def _run(api: _Api, key: str = "ctx7sk-test", waits: list | None = None) -> int:
    refresh = Context7Refresh(key=key, post=api, sleep=(waits if waits is not None else []).append)
    return refresh.run(library=LIBRARY, repository_url=URL, label="0.39.0")


@pytest.mark.fast
def test_refresh_starts_for_a_known_library(capsys):
    api = _Api((200, '{"message": "Refresh started successfully"}', {}))
    assert _run(api) == 0
    assert api.calls == [("/v1/refresh", {"libraryName": LIBRARY})]
    assert "started for 0.39.0" in capsys.readouterr().out


@pytest.mark.fast
def test_an_unknown_library_is_submitted_instead(capsys):
    api = _Api((404, '{"error": "library_not_found"}', {}), (200, '{"message": "Repository submitted"}', {}))
    assert _run(api) == 0
    assert api.calls[1] == ("/v2/add/repo/github", {"docsRepoUrl": URL})
    assert "submitted" in capsys.readouterr().out


@pytest.mark.fast
def test_the_ten_day_window_is_a_notice_not_a_failure(capsys):
    api = _Api((400, '{"error": "too-early", "message": "Minimum 10 days required between updates."}', {}))
    assert _run(api) == 0
    assert capsys.readouterr().out.startswith("::notice::")


@pytest.mark.fast
def test_a_rate_limit_is_waited_out_once():
    waits: list[int] = []
    api = _Api((429, "{}", {"Retry-After": "12"}), (200, "{}", {}))
    assert _run(api, waits=waits) == 0 and waits == [12] and len(api.calls) == 2

    # A wait too long to hold a release job for is a visible failure, not a silent skip.
    api = _Api((429, '{"error": "rate_limit_exceeded"}', {"Retry-After": "86400"}))
    assert _run(api, waits=waits) == 1 and waits == [12]


@pytest.mark.fast
@pytest.mark.parametrize("status", [401, 403, 500, 502])
def test_any_other_refusal_fails_visibly(capsys, status):
    assert _run(_Api((status, '{"error": "nope"}', {}))) == 1
    assert capsys.readouterr().out.startswith("::error::")


@pytest.mark.fast
def test_a_failed_submission_fails_and_a_missing_key_only_warns(capsys):
    api = _Api((404, '{"error": "library_not_found"}', {}), (403, '{"error": "forbidden"}', {}))
    assert _run(api) == 1
    capsys.readouterr()

    api = _Api()
    assert _run(api, key="") == 0 and api.calls == []
    assert capsys.readouterr().out.startswith("::warning::")
