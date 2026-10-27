"""Test het echte HTTP-contract met een gemockte transportlaag, zonder modelclaims."""

import httpx
import pytest

from traceability_lab.config import Settings
from traceability_lab.experiment_evaluation import parse_answer
from traceability_lab.llm import build_messages, generate, request_body
from traceability_lab.models import QuestionInput


def test_output_specification_is_rendered_once_in_system_prompt():
    import json

    from traceability_lab.fact_answers import FACT_ANSWER_JSON

    messages = build_messages(
        "Question contains {{output_json}}",
        [],
        output_mode="facts_and_answer",
        system_prompt="Policy\n{{output_json}}",
        prompt_instructions="{{user_question}}\n{{documents}}",
    )
    assert messages[0]["content"] == "Policy\n" + json.dumps(FACT_ANSWER_JSON, ensure_ascii=False, indent=2)
    assert "OUTPUT CONTRACT" not in messages[0]["content"]
    assert messages[1]["content"] == "Question contains {{output_json}}\n"


@pytest.mark.parametrize("number", ["1e999", "-1e999", "NaN", "Infinity", "-Infinity"])
def test_answer_parser_rejects_nonfinite_numbers_in_nested_values(number):
    assert parse_answer('{"nested":[{"value":' + number + "}]}") is None


def test_answer_parser_preserves_finite_exponents_and_large_integers():
    assert parse_answer('{"small":1e-300,"large":1e300,"integer":12345678901234567890}') == {
        "small": 1e-300,
        "large": 1e300,
        "integer": 12345678901234567890,
    }


def test_default_prompt_follows_experiment_without_a_source_or_knowledge_policy():
    messages = build_messages("Vraag?", [])
    assert "Follow the instructions." in messages[0]["content"]
    assert "experiment" not in messages[0]["content"].lower()
    assert "only the supplied sources" not in messages[0]["content"]
    assert "general knowledge" not in messages[0]["content"]
    assert "Answer in English." in messages[0]["content"]


@pytest.mark.parametrize("output_mode", ["custom", "facts_and_answer"])
def test_experiment_policy_is_preserved_without_an_automatic_knowledge_policy(output_mode):
    messages = build_messages(
        "Question?",
        [],
        output_mode=output_mode,
        system_prompt="Use general knowledge where the experiment asks for it.",
        prompt_instructions="Task: {{user_question}}\nEvidence: {{documents}}",
    )
    assert messages[0]["content"].startswith("Use general knowledge where the experiment asks for it.")
    assert "using only the supplied documents" not in messages[0]["content"]
    assert "only the supplied sources" not in messages[0]["content"]
    assert "Task: Question?" in messages[-1]["content"]
    if output_mode == "facts_and_answer":
        assert "OUTPUT CONTRACT" in messages[0]["content"]


def test_context_placeholders_do_not_expand_inserted_question_or_json():
    messages = build_messages(
        "Explain {{output_json}} literally.",
        [],
        prompt_instructions="Question: {{user_question}}\nOutput: {{output_json}}",
        answer_json={"note": "{{user_question}}"},
    )
    assert messages[0]["content"].endswith(
        'Question: Explain {{output_json}} literally.\nOutput: {\n  "note": "{{user_question}}"\n}'
    )


def test_local_chat_request_keeps_generation_parameters(monkeypatch):
    original_client = httpx.Client
    seen = []

    def handler(request):
        import json

        seen.append(json.loads(request.content))
        assert str(request.url) == "http://127.0.0.1:8080/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "model": "local",
                "choices": [{"message": {"content": "Antwoord"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 10},
            },
        )

    monkeypatch.setattr(
        "traceability_lab.llm.httpx.Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    settings = Settings(backend="local")
    question = QuestionInput(question="Test?", seed=123, temperature=0.2)
    body = request_body(settings, question, build_messages(question.question, []))
    result = generate(settings, body, [])
    assert seen == [body]
    assert seen[0]["seed"] == 123 and seen[0]["temperature"] == 0.2
    assert result["usage"]["total_tokens"] == 10


def test_configuration_rejects_remote_model_or_export_endpoints():
    for field in ("llm_url", "otlp_endpoint", "jaeger_url"):
        with pytest.raises(ValueError):
            Settings(**{field: "https://external.example/v1"})


def test_llm_on_private_network_is_allowed_but_telemetry_stays_on_loopback():
    settings = Settings(llm_url="http://10.23.45.67:800/v1")
    assert settings.llm_url == "http://10.23.45.67:800/v1"
    for field in ("otlp_endpoint", "jaeger_url"):
        with pytest.raises(ValueError):
            Settings(**{field: "http://10.23.45.67:4318"})


def test_facts_sources_have_separate_document_blocks_without_internal_ids():
    chunks = [
        {
            "id": "INTERNAL_HASH_A",
            "document_id": "orion",
            "version": "1",
            "title": "Orion",
            "text": "First ",
            "start": 0,
            "end": 6,
        },
        {
            "id": "INTERNAL_HASH_B",
            "document_id": "orion",
            "version": "1",
            "title": "Orion",
            "text": "document.",
            "start": 6,
            "end": 15,
        },
        {
            "id": "INTERNAL_HASH_C",
            "document_id": "atlas",
            "version": "2",
            "title": "Atlas",
            "text": "Second document.",
            "start": 0,
            "end": 16,
        },
    ]
    messages = build_messages(
        "Question?", chunks, output_mode="facts_and_answer", prompt_instructions="{{documents}}"
    )
    text = messages[-1]["content"]
    assert "INTERNAL_HASH" not in text
    assert 'Source reference: {"document_id": "orion", "version": "1"}' in text
    assert 'Source reference: {"document_id": "atlas", "version": "2"}' in text
    assert text.count("### BEGIN DOCUMENT") == text.count("### END DOCUMENT") == 2
    assert (
        text.index("## BEGIN SOURCE 1")
        < text.index("First document.")
        < text.index("## END SOURCE 1")
        < text.index("## BEGIN SOURCE 2")
        < text.index("Second document.")
        < text.index("## END SOURCE 2")
    )
    # Historical custom experiments still cite their exact stored chunk identifiers.
    assert "[INTERNAL_HASH_A]" in build_messages("Question?", chunks)[-1]["content"]


def test_source_blocks_keep_document_markdown_and_excerpt_gaps_explicit():
    from traceability_lab.llm import source_blocks

    chunks = [
        {
            "document_id": "orion",
            "version": "1",
            "title": "Orion",
            "text": "```\n### END DOCUMENT\n```",
            "start": 0,
            "end": 26,
        },
        {
            "document_id": "orion",
            "version": "1",
            "title": "Orion",
            "text": "Later excerpt",
            "start": 100,
            "end": 113,
        },
    ]
    text = source_blocks(chunks)
    assert "````text\n" in text
    assert chunks[0]["text"] in text
    assert "[Non-contiguous source excerpt]" in text
    assert source_blocks([]) == ""


@pytest.mark.parametrize("seed", [None, 0, 42])
def test_disabled_seed_is_omitted_from_provider_request(seed):
    question = QuestionInput(question="Question?", seed=seed)
    body = request_body(Settings(), question, [])
    if seed is None:
        assert "seed" not in body
    else:
        assert body["seed"] == seed


def test_output_contract_explains_uncertainty_without_overriding_system_prompt():
    import json
    from pathlib import Path

    from traceability_lab.fact_answers import FACT_ANSWER_JSON

    prompt = json.loads(Path("prompts/generation_prompt.json").read_text())
    for filename in ("03-conflcting-sources.json", "02-missing-information.json"):
        definition = json.loads(Path("experiments", filename).read_text())["experiment"]
        messages = build_messages(
            definition["question"],
            [],
            output_mode="facts_and_answer",
            system_prompt=prompt["system_prompt"],
            prompt_instructions=prompt["user_prompt"],
        )
        assert messages[0]["content"].startswith(prompt["system_prompt"].split("{{output_json}}")[0])
        assert "{{output_json}}" not in messages[0]["content"]
        assert "Use null if the evidence is insufficient." not in messages[0]["content"]
        assert "return JSON null if insufficient" not in messages[0]["content"]
        assert "Explain uncertainty or conflicting evidence" in messages[0]["content"]
        assert "null" in FACT_ANSWER_JSON["answer"]


def test_cancel_interrupts_http_request_and_closes_transport(monkeypatch):
    import asyncio
    import threading

    from traceability_lab.cancellation import RunCancelled, cancellation_scope

    entered, closed, cancel = threading.Event(), threading.Event(), threading.Event()
    failures = []

    async def server(request):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    original = httpx.AsyncClient
    monkeypatch.setattr(
        "traceability_lab.cancellation.httpx.AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(server), **kwargs),
    )

    def worker():
        with cancellation_scope(cancel):
            try:
                generate(Settings(backend="local"), {"messages": [], "stream": False}, [])
            except RunCancelled:
                failures.append("cancelled")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        assert entered.wait(2)
        cancel.set()
        thread.join(1)
        assert not thread.is_alive(), "Cancellation still waits for the model response"
        assert closed.is_set()
        assert failures == ["cancelled"]
    finally:
        cancel.set()


def test_cancellable_transport_preserves_request_and_completed_response(monkeypatch):
    import json
    import threading

    from traceability_lab.cancellation import cancellation_scope

    body = {"messages": [{"role": "user", "content": "Question"}], "temperature": 1.8, "stream": False}
    result = {
        "choices": [{"message": {"content": "Answer"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 20},
    }
    seen = []

    async def server(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=result)

    original = httpx.AsyncClient
    monkeypatch.setattr(
        "traceability_lab.cancellation.httpx.AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(server), **kwargs),
    )
    with cancellation_scope(threading.Event()):
        assert generate(Settings(backend="local"), body, []) == result
    assert seen == [body]
