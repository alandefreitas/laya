"""Remote mode of the MCP server (LAYA_BASE_URL): answers come from a laya-serve over HTTP.

A stdlib HTTP server stands in for laya-serve, so these tests need no checkpoint, no torch
and no network. Run: python -m pytest tests/test_mcp_remote.py
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("mcp")

from laya.mcp import server as mcp_server  # noqa: E402
from laya.mcp.remote import RemoteError, RemoteRouter  # noqa: E402
from laya.mcp.tools import ToolError, laya_predict, laya_predict_batch, laya_shortlist, laya_status  # noqa: E402

ANSWER = {
    "model": "laya-rl-agent",
    "answers": {"dept": {"type": "choice", "choice": "billing",
                         "probabilities": {"billing": 0.9, "tech": 0.1}, "confidence": 0.9}},
    "usage": {"input_tokens": 40, "output_tokens": 0},
    "routing": {"model": "english", "repo": "convaiinnovations/laya", "reason": "English Latin text"},
}
HEALTH = {"status": "ok", "loaded": ["english"], "revisions": {"english": "abc"},
          "device": "mps", "device_is_preference": False, "checkpoint_devices": {"english": "mps"}}


class FakeServe(BaseHTTPRequestHandler):
    """Records every request; answers /health and /v1/systemone like laya-serve."""

    requests: list = []
    status_override: int | None = None
    detail_override: str = "bad request"
    require_bearer: str | None = None

    def log_message(self, *args):  # quiet
        pass

    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        FakeServe.requests.append(("GET", self.path, None, dict(self.headers)))
        if self.path == "/health":
            if FakeServe.require_bearer and self.headers.get("Authorization") != "Bearer " + FakeServe.require_bearer:
                return self._send(200, {"status": "ok"})
            return self._send(200, HEALTH)
        self._send(404, {"detail": "Not Found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else None
        FakeServe.requests.append(("POST", self.path, body, dict(self.headers)))
        if FakeServe.require_bearer and self.headers.get("Authorization") != "Bearer " + FakeServe.require_bearer:
            return self._send(401, {"detail": "invalid or missing bearer token"})
        if FakeServe.status_override:
            return self._send(FakeServe.status_override, {"detail": FakeServe.detail_override})
        if self.path == "/v1/systemone":
            return self._send(200, ANSWER)
        self._send(404, {"detail": "Not Found"})


@pytest.fixture
def serve():
    FakeServe.requests = []
    FakeServe.status_override = None
    FakeServe.require_bearer = None
    httpd = HTTPServer(("127.0.0.1", 0), FakeServe)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def _questions():
    return {"dept": {"type": "choice", "instructions": "which team?", "criteria": {"billing": None, "tech": None}}}


def test_predict_posts_the_serve_body_and_returns_it_unchanged(serve):
    router = RemoteRouter(serve)
    result = router.predict({"text": "billed twice"}, _questions(), lang="en", max_len=512, min_confidence=0.3)
    assert result == ANSWER
    method, path, body, headers = FakeServe.requests[-1]
    assert (method, path) == ("POST", "/v1/systemone")
    assert body == {"state": {"text": "billed twice"}, "questions": _questions(),
                    "lang": "en", "max_len": 512, "min_confidence": 0.3}
    assert "Authorization" not in headers


def test_auto_model_is_not_sent_but_a_pinned_one_is(serve):
    router = RemoteRouter(serve)
    router.predict("x", _questions(), model="auto")
    assert "model" not in FakeServe.requests[-1][2]
    router.predict("x", _questions(), model="multilingual")
    assert FakeServe.requests[-1][2]["model"] == "multilingual"


def test_bearer_token_is_sent_when_configured(serve):
    FakeServe.require_bearer = "s3cret"
    router = RemoteRouter(serve, api_key="s3cret")
    assert router.predict("x", _questions()) == ANSWER
    assert FakeServe.requests[-1][3]["Authorization"] == "Bearer s3cret"
    with pytest.raises(RemoteError) as info:
        RemoteRouter(serve, api_key="wrong").predict("x", _questions())
    assert info.value.code == "unauthorized"


@pytest.mark.parametrize("status, code", [(400, "invalid_request"), (422, "invalid_request"),
                                          (503, "busy"), (500, "upstream_error")])
def test_http_errors_map_to_tool_error_codes(serve, status, code):
    FakeServe.status_override = status
    FakeServe.detail_override = "question 'dept': criteria must be an object"
    with pytest.raises(RemoteError) as info:
        RemoteRouter(serve).predict("x", _questions())
    assert info.value.code == code
    assert "criteria must be an object" in info.value.message


def test_unreachable_server_is_a_clean_error():
    with pytest.raises(RemoteError) as info:
        RemoteRouter("http://127.0.0.1:9", timeout=2).predict("x", _questions())
    assert info.value.code == "unreachable"


def test_loaded_and_revisions_come_from_health(serve):
    router = RemoteRouter(serve)
    assert router.loaded == ["english"]
    assert router.loaded_revisions == {"english": "abc"}
    assert RemoteRouter("http://127.0.0.1:9", timeout=2).loaded == []


def test_routing_stays_local_and_needs_no_server():
    router = RemoteRouter("http://127.0.0.1:9", timeout=2)
    decision = router.route({"text": "Mein Konto wurde zweimal belastet"}, _questions())
    assert decision.model == "multilingual"
    assert FakeServe.requests == [] or all(r[0] != "POST" for r in FakeServe.requests)


def test_predict_batch_is_one_call_per_request_in_order(serve):
    router = RemoteRouter(serve)
    items = [{"state": "a", "questions": _questions()},
             {"state": "b", "questions": _questions(), "model": "english", "max_len": 256}]
    results = router.predict_batch(items, batch_size=8, sort_by_length=True, min_confidence=0.5)
    assert results == [ANSWER, ANSWER]
    posts = [r for r in FakeServe.requests if r[0] == "POST"]
    assert [p[2]["state"] for p in posts] == ["a", "b"]
    assert posts[1][2]["model"] == "english" and posts[1][2]["max_len"] == 256
    assert all(p[2]["min_confidence"] == 0.5 for p in posts)


def test_lifecycle_is_server_owned(serve):
    router = RemoteRouter(serve)
    assert router.preload(["english"]) is router
    assert router.unload() is None
    with pytest.raises(RemoteError) as info:
        router.load("english")
    assert info.value.code == "unsupported_remote"


def test_tools_translate_remote_errors(serve):
    FakeServe.status_override = 422
    FakeServe.detail_override = "question 'dept': unknown type"
    with pytest.raises(ToolError) as info:
        laya_predict({"text": "x"}, _questions(), router=RemoteRouter(serve))
    assert info.value.code == "invalid_request" and "unknown type" in info.value.message
    with pytest.raises(ToolError) as info:
        laya_predict_batch([{"state": {"text": "x"}, "questions": _questions()}], router=RemoteRouter(serve))
    assert info.value.code == "invalid_request"


def test_predict_tool_shape_through_a_remote_router(serve):
    out = laya_predict({"text": "billed twice"}, _questions(), router=RemoteRouter(serve))
    assert out["answers"] == ANSWER["answers"]
    assert out["routing"]["model"] == "english"
    assert "device" not in out          # no in-process agent to read a device from
    assert out["latency_ms"] >= 0


def test_status_reports_the_server_not_local_torch(serve):
    out = laya_status(router=RemoteRouter(serve), preload=False)
    assert out["mode"] == "remote" and out["base_url"] == serve
    assert out["loaded"] == ["english"] and out["device"] == "mps"
    assert out["server"]["status"] == "ok"
    assert "torch_version" not in out
    down = laya_status(router=RemoteRouter("http://127.0.0.1:9", timeout=2))
    assert down["mode"] == "remote" and down["loaded"] == [] and "server_error" in down


def test_shortlist_is_refused_in_remote_mode(serve):
    many = {"pick": {"type": "choice", "instructions": "pick", "criteria": {"opt%d" % i: None for i in range(60)}}}
    with pytest.raises(ToolError) as info:
        laya_shortlist({"text": "x"}, many, k=5, router=RemoteRouter(serve))
    assert info.value.code == "unsupported_remote"


def test_server_builds_a_remote_router_from_the_environment(monkeypatch, serve):
    monkeypatch.setenv("LAYA_BASE_URL", serve + "/")
    monkeypatch.setenv("LAYA_API_KEY", "k")
    monkeypatch.setattr(mcp_server, "_ROUTER", None)
    router = mcp_server._ensure_router()
    assert isinstance(router, RemoteRouter)
    assert router.base_url == serve and router.api_key == "k"
    monkeypatch.setattr(mcp_server, "_ROUTER", None)


@pytest.mark.parametrize("raw, expected", [("", None), ("  ", None), ("127.0.0.1:8123", "http://127.0.0.1:8123"),
                                           ("http://h:1/", "http://h:1"), ("https://laya.example", "https://laya.example")])
def test_base_url_normalisation(monkeypatch, raw, expected):
    monkeypatch.setenv("LAYA_BASE_URL", raw)
    assert mcp_server._remote_base_url() == expected
