from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path


class IngestResponseError(RuntimeError):
    """The cloud did not acknowledge the exact event that was sent."""


def validate_ingest_response(
    response: dict[str, object],
    expected_event_id: str,
    *,
    require_remote_image: bool = True,
) -> dict[str, object]:
    if response.get("ok") is not True:
        raise IngestResponseError("ingest response did not report ok=true")
    response_event_id = response.get("event_id")
    if not isinstance(response_event_id, str) or response_event_id != expected_event_id:
        raise IngestResponseError("ingest response event_id does not match the sent event")
    remote_id = response.get("id")
    if isinstance(remote_id, bool):
        raise IngestResponseError("ingest response id must be a positive integer")
    if isinstance(remote_id, int):
        valid_remote_id = remote_id > 0
    elif isinstance(remote_id, str):
        valid_remote_id = remote_id.isdigit() and int(remote_id) > 0
    else:
        valid_remote_id = False
    if not valid_remote_id:
        raise IngestResponseError("ingest response id must be a positive integer")
    if require_remote_image:
        core_pair = (response.get("core_image_url"), response.get("core_image_public_id"))
        legacy_pair = (response.get("image_url"), response.get("image_public_id"))
        if not any(
            all(isinstance(value, str) and value.strip() for value in pair)
            for pair in (core_pair, legacy_pair)
        ):
            raise IngestResponseError(
                "ingest response must confirm persisted core-weight evidence"
            )
    return response


def post_measurement(
    url: str,
    payload: dict[str, object],
    image_path: str | Path,
    token: str,
    timeout: float = 10.0,
) -> dict[str, object]:
    body = dict(payload)
    body["image_base64"] = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    # The single captured frame is specifically the core-weight evidence image.
    # Keep the established image_base64 field so older deployed Edge Functions
    # remain compatible while newer functions persist it under core_image_*.
    body["image_role"] = "core_weight"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Device-Token": token,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_body = response.read()
        if not 200 <= response.status < 300:
            raise RuntimeError(f"API returned HTTP {response.status}")
    if not response_body:
        return {}
    parsed = json.loads(response_body.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise RuntimeError("API response must be a JSON object")
    return parsed
