from __future__ import annotations

import base64
import http.client
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


UPSTREAM_HOST = "127.0.0.1"
UPSTREAM_PORT = 8080
MAX_BODY_BYTES = 32 * 1024 * 1024
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


class AuthProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        return supplied == self.server.expected_authorization

    def _authenticate(self) -> None:
        body = b"Authentication required"
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Tram Can QR Pilot"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self) -> None:
        if not self._authorized():
            self._authenticate()
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self.send_error(413, "Request body too large")
            return
        body = self.rfile.read(length) if length else None
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in HOP_BY_HOP
            and name.lower() not in {"authorization", "host", "content-length"}
        }
        if body is not None:
            headers["Content-Length"] = str(len(body))
        headers["Host"] = f"{UPSTREAM_HOST}:{UPSTREAM_PORT}"

        connection = http.client.HTTPConnection(UPSTREAM_HOST, UPSTREAM_PORT, timeout=30)
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read()
            self.send_response(response.status, response.reason)
            for name, value in response.getheaders():
                if name.lower() not in HOP_BY_HOP and name.lower() != "content-length":
                    self.send_header(name, value)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        except (OSError, http.client.HTTPException):
            self.send_error(502, "Local pilot server unavailable")
        finally:
            connection.close()

    do_GET = _proxy
    do_POST = _proxy
    do_OPTIONS = _proxy

    def log_message(self, format: str, *args: object) -> None:
        return


class AuthProxyServer(ThreadingHTTPServer):
    expected_authorization: str


def main() -> None:
    user = os.environ.get("ROLL_SCALE_PILOT_USER", "pilot")
    password = os.environ.get("ROLL_SCALE_PILOT_PASSWORD", "")
    if not password:
        raise ValueError("ROLL_SCALE_PILOT_PASSWORD is required")
    token = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
    server = AuthProxyServer(("127.0.0.1", 8081), AuthProxyHandler)
    server.expected_authorization = f"Basic {token}"
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
