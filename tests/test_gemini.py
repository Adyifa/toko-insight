import json

import pytest

from tokoinsight.gemini import GeminiClient, GeminiError, retry_delay


def make(responses):
    sent = []

    def transport(url, headers, body):
        sent.append((url, headers, json.loads(body)))
        status, payload = responses.pop(0)
        return status, json.dumps(payload).encode()

    return GeminiClient(api_key="test", transport=transport, sleep=lambda s: None), sent


def test_function_call_request_and_response():
    content = {"role": "model", "parts": [{"functionCall": {"name": "top_products", "args": {"n": 3}}}]}
    client, sent = make([(200, {"candidates": [{"content": content}], "usageMetadata": {"promptTokenCount": 10}})])
    result = client.generate([{"role": "user", "parts": [{"text": "hi"}]}], system="sys",
                             tools=[{"name": "top_products", "description": "d"}])
    assert result == content
    url, headers, payload = sent[0]
    assert url.endswith("models/gemini-2.5-flash:generateContent") and headers["x-goog-api-key"] == "test"
    assert payload["tools"][0]["functionDeclarations"][0]["name"] == "top_products"
    assert payload["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
    assert payload["systemInstruction"]["parts"][0]["text"] == "sys"
    assert payload["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0
    assert client.usage.calls == 1 and client.usage.input_tokens == 10


def test_retries_rate_limit_then_succeeds():
    ok = {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}
    limited = {"error": {"message": "quota", "details": [{"retryDelay": "3s"}]}}
    client, sent = make([(429, limited), (503, {}), (200, ok)])
    assert client.generate([])["parts"][0]["text"] == "hi"
    assert len(sent) == 3
    assert retry_delay(limited, 0) == 3.5


def test_errors():
    client, _ = make([(400, {"error": {"message": "bad request"}})])
    with pytest.raises(GeminiError, match="400"):
        client.generate([])
    client, _ = make([(200, {"candidates": [{"content": {}, "finishReason": "MALFORMED_FUNCTION_CALL"}]})])
    with pytest.raises(GeminiError, match="MALFORMED"):
        client.generate([])
    with pytest.raises(GeminiError, match="GEMINI_API_KEY"):
        GeminiClient(api_key="")
