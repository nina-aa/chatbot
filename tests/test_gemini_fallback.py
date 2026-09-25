import pytest
from google.genai import errors

from docrag.llm.base import AllModelsFailed
from docrag.llm.gemini_llm import GeminiLLM


class FakeClient:
    """outcomes maps model -> HTTP code to raise, or a string to return."""

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.tried = []
        self.configs = {}
        self.models = self

    def generate_content(self, model, contents, config):
        self.tried.append(model)
        self.configs[model] = (contents, config)
        outcome = self.outcomes[model]
        if isinstance(outcome, int):
            raise errors.ClientError(outcome, {"error": {"message": "x"}})
        return type("R", (), {"text": outcome})()


def test_429_moves_straight_to_next_model():
    client = FakeClient({"a": 429, "b": 429, "c": "answer"})
    result = GeminiLLM(["a", "b", "c"], client=client).generate("q")
    assert result.text == "answer" and result.model == "c"
    assert client.tried == ["a", "b", "c"]


def test_stops_after_first_success():
    client = FakeClient({"a": "ok", "b": "never"})
    GeminiLLM(["a", "b"], client=client).generate("q")
    assert client.tried == ["a"]


def test_404_is_skipped_too():
    client = FakeClient({"ghost": 404, "b": "ok"})
    assert GeminiLLM(["ghost", "b"], client=client).generate("q").model == "b"


def test_all_failing_reports_each_model():
    client = FakeClient({"a": 429, "b": 503})
    with pytest.raises(AllModelsFailed) as e:
        GeminiLLM(["a", "b"], client=client).generate("q")
    assert e.value.errors == {"a": "HTTP 429", "b": "HTTP 503"}


def test_other_errors_are_not_swallowed():
    client = FakeClient({"a": 400, "b": "ok"})
    with pytest.raises(errors.ClientError):
        GeminiLLM(["a", "b"], client=client).generate("q")
    assert client.tried == ["a"]


def test_empty_response_falls_through():
    client = FakeClient({"a": "", "b": "ok"})
    assert GeminiLLM(["a", "b"], client=client).generate("q").model == "b"


def test_gemma_gets_system_prompt_inside_user_content():
    client = FakeClient({"gemma-3-12b-it": "ok"})
    GeminiLLM(["gemma-3-12b-it"], client=client).generate("question", system="be brief")
    contents, config = client.configs["gemma-3-12b-it"]
    assert contents.startswith("be brief") and config.system_instruction is None


def test_json_mode_is_requested_for_gemini_but_not_for_gemma():
    client = FakeClient({"gemini-x": "{}", "gemma-3-12b-it": "{}"})
    GeminiLLM(["gemini-x"], client=client).generate("q", json_mode=True)
    GeminiLLM(["gemma-3-12b-it"], client=client).generate("q", json_mode=True)
    assert client.configs["gemini-x"][1].response_mime_type == "application/json"
    assert client.configs["gemma-3-12b-it"][1].response_mime_type is None


def test_server_errors_and_timeouts_are_skipped_not_fatal():
    # 500 crashed a whole eval run once; 504 is what an expired request timeout comes back as.
    for code in (500, 502, 503, 504):
        client = FakeClient({"a": code, "b": "ok"})
        assert GeminiLLM(["a", "b"], client=client).generate("q").model == "b", code


def test_a_chain_of_broken_models_is_reported_not_raised_as_a_server_error():
    client = FakeClient({"a": 500, "b": 504})
    with pytest.raises(AllModelsFailed) as e:
        GeminiLLM(["a", "b"], client=client).generate("q")
    assert e.value.errors == {"a": "HTTP 500", "b": "HTTP 504"}


def test_the_client_is_built_with_a_request_timeout(monkeypatch):
    seen = {}
    monkeypatch.setattr("docrag.llm.gemini_llm.genai.Client", lambda **kw: seen.update(kw) or object())
    GeminiLLM(["a"], api_key="k", timeout_s=45)
    assert seen["http_options"].timeout == 45_000
    GeminiLLM(["a"], api_key="k")
    assert seen["http_options"].timeout == 120_000
