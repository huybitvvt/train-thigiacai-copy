from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request


def lookup_roll(
    url: str,
    qr_code: str,
    token: str,
    timeout: float = 10.0,
) -> dict[str, object]:
    separator = "&" if "?" in url else "?"
    request_url = f"{url}{separator}{urllib.parse.urlencode({'qr': qr_code})}"
    request = urllib.request.Request(
        request_url,
        headers={"Accept": "application/json", "X-Device-Token": token},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError(f"Lookup API returned HTTP {exc.code}") from exc
        if exc.code == 404 and isinstance(parsed, dict):
            return parsed
        error = parsed.get("error", "unknown") if isinstance(parsed, dict) else "unknown"
        raise RuntimeError(f"Lookup API returned HTTP {exc.code}: {error}") from exc

    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Lookup API returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Lookup API response must be a JSON object")
    return parsed
