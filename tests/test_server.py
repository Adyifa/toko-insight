import json
import threading
import urllib.error
import urllib.request

import pytest
from conftest import FakeLLM, call, text

from tokoinsight.agent import Agent
from tokoinsight.server import make_server


@pytest.fixture()
def serve(db, toolbox):
    servers = []

    def start(agent=None):
        server = make_server(db, agent, port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def request(url, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def test_page_and_dashboard(serve):
    base = serve()
    status, page = request(base + "/")
    assert status == 200 and b"Toko Insight" in page
    status, body = request(base + "/api/dashboard")
    data = json.loads(body)
    assert status == 200 and data["agent"] is False
    assert data["top"]["rows"][0]["product"] == "Beras Premium 5kg"
    status, body = request(base + "/api/forecast?product=Kurma&horizon=6")
    assert status == 200 and len(json.loads(body)["forecast"]) == 6
    status, body = request(base + "/api/forecast?product=laptop")
    assert status == 400


def test_sql_endpoint_is_read_only(serve):
    base = serve()
    status, body = request(base + "/api/sql", {"query": "SELECT COUNT(*) AS n FROM stores"})
    assert status == 200 and json.loads(body)["rows"] == [[3]]
    status, body = request(base + "/api/sql", {"query": "DROP TABLE stores"})
    assert status == 400 and "Only SELECT" in json.loads(body)["error"]


def test_ask_without_key_explains(serve):
    status, body = request(serve() + "/api/ask", {"question": "hi"})
    assert status == 503 and "GEMINI_API_KEY" in json.loads(body)["error"]


def test_ask_with_agent(serve, toolbox):
    agent = Agent(FakeLLM([call("purchase_frequency", n=3), text("Payung Lipat is bought least.")]), toolbox)
    status, body = request(serve(agent) + "/api/ask", {"question": "rarest?", "history": []})
    data = json.loads(body)
    assert status == 200 and data["answer"].startswith("Payung")
    assert data["steps"][0]["name"] == "purchase_frequency" and data["steps"][0]["sql"]
