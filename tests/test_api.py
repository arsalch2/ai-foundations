"""Offline tests: a fake client stands in for the LLM, so these run in CI with no API key."""

import json
from types import SimpleNamespace as NS

from fastapi.testclient import TestClient

from app.llm import get_client
from app.main import app
from app.schemas import OrderExtraction
from evals.run_eval import score


def tool_call(call_id: str, name: str, args: dict):
    return NS(id=call_id, type="function", function=NS(name=name, arguments=json.dumps(args)))


class FakeMessage(NS):
    def model_dump(self, exclude_none: bool = False) -> dict:
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in (self.tool_calls or [])
            ],
        }


class FakeCompletions:
    def __init__(self, scripted: list | None = None):
        self.scripted = scripted or []
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(
                [NS(choices=[NS(delta=NS(content=t))]) for t in ["Hal", "lo", "!"]] + [NS(choices=[])]
            )
        if self.scripted:
            msg = self.scripted.pop(0)
        else:
            msg = FakeMessage(content="Hallo!", tool_calls=None)
        return NS(choices=[NS(message=msg)], model="fake-model", usage=NS(prompt_tokens=7, completion_tokens=2))

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        parsed = OrderExtraction(
            is_order=True,
            customer_company="Becker Antriebstechnik GmbH",
            customer_email=None,
            items=[{"product": "6205-2RS", "quantity": 200, "unit": "Stück"}],
            requested_delivery_date="2027-01-15",
            language="de",
        )
        return NS(choices=[NS(message=NS(parsed=parsed, refusal=None))])


def make_client(completions: FakeCompletions) -> TestClient:
    fake = NS(chat=NS(completions=completions))
    app.dependency_overrides[get_client] = lambda: fake
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_chat_returns_answer_and_usage():
    client = make_client(FakeCompletions())
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "Hi"}]})
    assert r.status_code == 200
    assert r.json() == {"answer": "Hallo!", "model": "fake-model", "prompt_tokens": 7, "completion_tokens": 2}


def test_stream_emits_tokens_then_done():
    client = make_client(FakeCompletions())
    r = client.post("/chat/stream", json={"messages": [{"role": "user", "content": "Hi"}]})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [b for b in r.text.split("\n\n") if b]
    assert [e.splitlines()[0] for e in events] == ["event: token"] * 3 + ["event: done"]
    text = "".join(json.loads(e.splitlines()[1][6:])["text"] for e in events[:3])
    assert text == "Hallo!"


def test_extract_wraps_email_as_untrusted_data():
    completions = FakeCompletions()
    client = make_client(completions)
    r = client.post("/extract/order", json={"email_text": "bitte 200 Stück 6205-2RS"})
    assert r.status_code == 200
    assert r.json()["result"]["items"][0]["quantity"] == 200
    user_msg = completions.calls[0]["messages"][1]["content"]
    assert user_msg.startswith("<email>") and user_msg.endswith("</email>")
    assert completions.calls[0]["response_format"] is OrderExtraction


def test_ask_runs_tool_then_answers():
    completions = FakeCompletions(
        scripted=[
            FakeMessage(content=None, tool_calls=[tool_call("c1", "CheckStock", {"article_number": "6205-2rs"})]),
            FakeMessage(content="Ja, 1240 Stück auf Lager.", tool_calls=None),
        ]
    )
    client = make_client(completions)
    r = client.post("/ask", json={"question": "Haben Sie 200 Stück 6205-2RS?"})
    body = r.json()
    assert r.status_code == 200
    assert body["tool_calls"][0]["result"]["in_stock"] == 1240
    assert body["answer"].startswith("Ja")
    # the tool result was sent back to the model
    assert completions.calls[1]["messages"][-1]["role"] == "tool"


def test_ask_stops_runaway_tool_loops():
    looping = [FakeMessage(content=None, tool_calls=[tool_call(f"c{i}", "CheckStock", {"article_number": "X"})]) for i in range(10)]
    client = make_client(FakeCompletions(scripted=looping))
    r = client.post("/ask", json={"question": "loop"})
    assert r.status_code == 508


def test_ask_handles_bad_tool_arguments():
    bad = NS(id="c1", type="function", function=NS(name="CheckStock", arguments="{not json"))
    client = make_client(
        FakeCompletions(scripted=[FakeMessage(content=None, tool_calls=[bad]), FakeMessage(content="Sorry.", tool_calls=None)])
    )
    r = client.post("/ask", json={"question": "?"})
    assert r.status_code == 200
    assert "error" in r.json()["tool_calls"][0]["result"]


def test_eval_scoring_ignores_case_and_item_order():
    expected = {
        "is_order": True,
        "customer_company": "Becker GmbH",
        "items": [{"product": "6205-2RS", "quantity": 2}, {"product": "6305-ZZ", "quantity": 1}],
        "requested_delivery_date": None,
        "language": "de",
    }
    got = {**expected, "customer_company": "becker  gmbh", "items": list(reversed(expected["items"]))}
    assert all(score(expected, got).values())
