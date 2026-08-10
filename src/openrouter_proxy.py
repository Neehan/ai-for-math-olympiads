"""Loopback Anthropic-API shim that freezes OpenRouter provider routing.

Claude Agent SDK does not expose OpenRouter's top-level ``provider`` request
field.  This tiny same-container proxy inserts the repository-defined route,
then streams the otherwise unchanged Anthropic Messages response.  It binds to
loopback only and rejects model substitution.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from src.openrouter_routing import route_for

log = logging.getLogger("openrouter_proxy")

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8787
UPSTREAM_HOST = "openrouter.ai"
ALLOWED_MODEL_ENV = "HARNESS_OPENROUTER_ALLOWED_MODEL"
MESSAGES_PATH = "/api/v1/messages"
MAX_REQUEST_BYTES = 128 * 1024 * 1024

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def routed_body(raw: bytes, allowed_model: str) -> bytes:
    """Validate the requested model and insert its frozen upstream route."""
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("request body is not valid JSON") from error
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    requested_model = body.get("model")
    if requested_model != allowed_model:
        raise ValueError(
            f"model substitution rejected: expected {allowed_model!r}, "
            f"received {requested_model!r}"
        )
    route = route_for(allowed_model)
    if route is None:
        # Ordinary OpenRouter model IDs need no body transformation, but are
        # still locked to the controller-selected model.
        return raw
    body.update(route)
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def upstream_path(path: str) -> str:
    """Remove Claude CLI's Anthropic-only beta query flag for OpenRouter."""
    parsed = urlsplit(path)
    query_items = [
        (name, value)
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
        if name != "beta"
    ]
    query = urlencode(query_items)
    return urlunsplit(("", "", parsed.path, query, ""))


def is_messages_path(path: str) -> bool:
    """Accept only OpenRouter's Anthropic Messages compatibility endpoint."""
    return urlsplit(path).path == MESSAGES_PATH


class RoutingProxyHandler(BaseHTTPRequestHandler):
    """Forward one request while streaming its response to Claude CLI."""

    protocol_version = "HTTP/1.1"
    server_version = "OlympiadOpenRouterShim/1"
    allowed_model: ClassVar[str]

    def log_message(self, format: str, *args: object) -> None:
        log.info("%s - %s", self.address_string(), format % args)

    def _send_json_error(self, status: int, message: str) -> None:
        payload = json.dumps({"error": {"message": message}}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path == "/health":
            payload = b"ok\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self._send_json_error(405, "only Anthropic Messages POST requests are allowed")

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
        # Claude CLI probes its configured base URL before its first request.
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        downstream_started = False
        if not is_messages_path(self.path):
            self._send_json_error(405, "only Anthropic Messages POST requests are allowed")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_json_error(411, "a valid Content-Length is required")
            return
        if not 0 < length <= MAX_REQUEST_BYTES:
            self._send_json_error(413, "request body size is invalid or too large")
            return
        try:
            body = routed_body(self.rfile.read(length), self.allowed_model)
        except ValueError as error:
            self._send_json_error(400, str(error))
            return

        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in _HOP_BY_HOP
            and name.lower() not in {"host", "content-length", "accept-encoding"}
        }
        headers["Content-Length"] = str(len(body))
        # Keep upstream errors human-readable and avoid forwarding a compressed
        # body after http.client has already normalized transfer framing.
        headers["Accept-Encoding"] = "identity"
        headers["X-OpenRouter-Metadata"] = "enabled"
        connection = http.client.HTTPSConnection(UPSTREAM_HOST, timeout=3_700)
        try:
            connection.request(
                "POST", upstream_path(self.path), body=body, headers=headers
            )
            upstream = connection.getresponse()
            if upstream.status >= 400:
                error_body = upstream.read()
                log.error(
                    "OpenRouter returned %s for %s",
                    upstream.status,
                    upstream_path(self.path),
                )
                self.send_response(upstream.status, upstream.reason)
                self.send_header(
                    "Content-Type",
                    upstream.getheader("Content-Type", "application/json"),
                )
                self.send_header("Content-Length", str(len(error_body)))
                self.send_header("Connection", "close")
                self.end_headers()
                downstream_started = True
                self.wfile.write(error_body)
                return
            self.send_response(upstream.status, upstream.reason)
            for name, value in upstream.getheaders():
                if name.lower() not in _HOP_BY_HOP and name.lower() != "content-length":
                    self.send_header(name, value)
            # http.client dechunks upstream data. Closing the downstream
            # connection delimits the streamed response without buffering it.
            self.send_header("Connection", "close")
            self.end_headers()
            downstream_started = True
            while chunk := upstream.read(65_536):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException) as error:
            if not downstream_started:
                self._send_json_error(502, f"OpenRouter transport failed: {error}")
            else:
                self.close_connection = True
        finally:
            connection.close()
            self.close_connection = True

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    allowed_model = os.environ.get(ALLOWED_MODEL_ENV, "").strip()
    if not allowed_model:
        raise SystemExit(f"{ALLOWED_MODEL_ENV} is required")
    RoutingProxyHandler.allowed_model = allowed_model
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), RoutingProxyHandler)
    server.daemon_threads = True
    log.info("OpenRouter routing shim listening for model %s", allowed_model)
    server.serve_forever()


if __name__ == "__main__":
    main()
