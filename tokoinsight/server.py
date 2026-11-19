"""Small JSON + static web server for the dashboard and chat (standard library only).

GET  /                      dashboard + chat page (web/index.html)
GET  /api/dashboard         KPIs, monthly trend, top/bottom products, ABC, growth, seasonal products
GET  /api/products          stores, categories, products
GET  /api/forecast?product=&category=&horizon=   backtested forecast
GET  /api/restock           restock plan
GET  /api/pairs?product=    products bought together
POST /api/ask   {"question": str, "history": [[q, a], ...]}  -> agent answer with tool calls and SQL
POST /api/sql   {"query": str}                               -> read-only SQL result
"""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

from . import analytics
from .analytics import AnalyticsError
from .forecasting import forecast
from .restock import restock_plan
from .sqlguard import ReadOnlySQL, SQLRejected

MAX_BODY = 64 * 1024


def dashboard(db) -> dict:
    return {
        "overview": analytics.overview(db),
        "trend": analytics.sales_trend(db, granularity="month", metric="revenue"),
        "top": analytics.top_products(db, n=10),
        "bottom": analytics.top_products(db, n=10, order="bottom"),
        "abc": analytics.abc_analysis(db)["summary"],
        "growth": analytics.growth(db, n=5),
        "categories": analytics.growth(db, level="category", n=25),
        "seasonality": analytics.seasonality(db),
        "pairs": analytics.basket_pairs(db, n=8),
    }


def make_handler(db, agent=None):
    lock = threading.Lock()  # one SQLite connection and one rate-limited API key: one request at a time
    page = resources.files("tokoinsight").joinpath("web/index.html").read_bytes()
    sql = ReadOnlySQL(db.path)
    cache: dict[str, dict] = {}

    def cached(key: str, func):
        if key not in cache:
            cache[key] = func()
        return cache[key]

    class Handler(BaseHTTPRequestHandler):
        server_version = "TokoInsight/0.1"

        def log_message(self, format: str, *args) -> None:  # keep the console quiet
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data) -> None:
            body = json.dumps(data, ensure_ascii=False, default=str).encode()
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                with lock:
                    if url.path in ("/", "/index.html"):
                        return self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
                    if url.path == "/api/dashboard":
                        data = cached("dashboard", lambda: dashboard(db))
                        data = {**data, "agent": bool(agent)}
                    elif url.path == "/api/products":
                        data = cached("products", lambda: analytics.list_products(db))
                    elif url.path == "/api/forecast":
                        args = {k: q[k] for k in ("product", "category", "store") if q.get(k)}
                        args["horizon"] = int(q.get("horizon", 6))
                        data = cached("fc" + json.dumps(args, sort_keys=True), lambda: forecast(db, **args))
                    elif url.path == "/api/restock":
                        data = cached("restock" + q.get("store", ""), lambda: restock_plan(db, store=q.get("store")))
                    elif url.path == "/api/pairs":
                        data = cached("pairs" + q.get("product", ""),
                                      lambda: analytics.basket_pairs(db, product=q.get("product"), n=15))
                    else:
                        return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except (AnalyticsError, ValueError) as exc:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            self._json(HTTPStatus.OK, data)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "empty or too large request"})
            try:
                data = json.loads(self.rfile.read(length))
            except ValueError:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
            if self.path == "/api/sql":
                try:
                    with lock:
                        return self._json(HTTPStatus.OK, sql.run(str(data.get("query", ""))))
                except SQLRejected as exc:
                    return self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            if self.path != "/api/ask":
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            if agent is None:
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {
                    "error": "The chat needs a Gemini API key. Put GEMINI_API_KEY=... in .env (free key at "
                             "https://aistudio.google.com/apikey) and restart `python -m tokoinsight serve`."})
            try:
                question = str(data.get("question", "")).strip()[:1000]
                history = [(str(a), str(b)) for a, b in data.get("history", [])][-6:]
            except (TypeError, ValueError, AttributeError):
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid history"})
            if not question:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "question is empty"})
            try:
                with lock:
                    answer = agent.ask(question, history)
            except Exception as exc:  # noqa: BLE001 - report API errors to the page
                return self._json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
            self._json(HTTPStatus.OK, answer.to_dict())

    return Handler


def make_server(db, agent=None, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(db, agent))


def serve(db, agent=None, host: str = "127.0.0.1", port: int = 8000) -> None:
    server = make_server(db, agent, host, port)
    print(f"Toko Insight running at http://{host}:{server.server_address[1]}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
