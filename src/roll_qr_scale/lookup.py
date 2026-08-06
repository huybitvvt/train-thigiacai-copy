from __future__ import annotations

import argparse
import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from .lookup_client import lookup_roll
from .qr_reader import QRReader


def _default_lookup_url() -> str | None:
    configured = os.environ.get("ROLL_SCALE_LOOKUP_URL")
    if configured:
        return configured
    ingest_url = os.environ.get("ROLL_SCALE_API_URL")
    if ingest_url and ingest_url.rstrip("/").endswith("/ingest-measurement"):
        return ingest_url.rstrip("/").removesuffix("/ingest-measurement") + "/lookup-roll"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Look up the latest confirmed weight by QR")
    parser.add_argument("--api-url", default=_default_lookup_url())
    parser.add_argument(
        "--api-token",
        default=os.environ.get("ROLL_SCALE_LOOKUP_TOKEN")
        or os.environ.get("ROLL_SCALE_DEVICE_TOKEN")
        or os.environ.get("ROLL_SCALE_API_TOKEN"),
    )
    parser.add_argument("--qr", help="Look up one QR and exit")
    parser.add_argument("--image", help="Decode a QR from an image, look it up, then exit")
    parser.add_argument("--serve", action="store_true", help="Open a local scanner-friendly web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser


def _format_result(result: dict[str, object]) -> str:
    if not result.get("found"):
        return f"KHÔNG TÌM THẤY QR: {result.get('qr_code', '')}"
    measurement = result["measurement"]
    assert isinstance(measurement, dict)
    gross = measurement.get("gross_weight", measurement.get("weight"))
    tare = measurement.get("tare_weight", 0)
    net = measurement.get("net_weight", gross)
    return (
        f"QR={measurement.get('qr_code')} | GROSS={gross} {measurement.get('unit')} | "
        f"TARE={tare} {measurement.get('unit')} | NET={net} {measurement.get('unit')} | "
        f"TIME={measurement.get('captured_at')} | HISTORY={result.get('history_count', 1)}"
    )


def _lookup_from_image(path: str) -> str:
    frame = cv2.imread(str(Path(path)))
    if frame is None:
        raise ValueError(f"Cannot read image: {path}")
    detections = QRReader().decode(frame)
    if not detections:
        raise ValueError("No QR found in image")
    return detections[0].value


LOOKUP_HTML = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tra cứu cân theo QR</title>
<style>
body{font-family:Arial,sans-serif;background:#f4f6f8;margin:0;padding:24px;color:#17202a}
.card{max-width:820px;margin:auto;background:white;padding:24px;border-radius:14px;box-shadow:0 4px 20px #0002}
h1{margin-top:0} form{display:flex;gap:10px} input{flex:1;font-size:22px;padding:12px}
button{font-size:18px;padding:10px 18px}.result{margin-top:22px;padding:18px;border-radius:10px;background:#eef2f5}
.weight{font-size:36px;font-weight:bold;color:#126b38}.error{color:#b00020;font-weight:bold} img{max-width:100%;margin-top:16px;border-radius:8px}
</style></head><body><div class="card"><h1>Tra cứu cân theo QR</h1>
<form id="form"><input id="qr" autocomplete="off" autofocus placeholder="Quét QR bằng scanner USB"><button>Tra cứu</button></form>
<div id="result" class="result">Đang chờ quét QR…</div></div>
<script>
const form=document.getElementById('form'), input=document.getElementById('qr'), result=document.getElementById('result');
form.addEventListener('submit',async e=>{e.preventDefault();const qr=input.value.trim();if(!qr)return;
 result.textContent='Đang tra cứu…';
 try{const response=await fetch('/api/lookup?qr='+encodeURIComponent(qr));const data=await response.json();
  if(!data.found){result.innerHTML='<div class="error">Không tìm thấy QR</div>';}
  else{const m=data.measurement;result.textContent='';
   const title=document.createElement('div');title.textContent='QR: '+m.qr_code;result.appendChild(title);
   const w=document.createElement('div');w.className='weight';w.textContent='NET: '+m.net_weight+' '+m.unit;result.appendChild(w);
   const details=document.createElement('div');details.textContent='Gross: '+m.gross_weight+' '+m.unit+' | Tare: '+m.tare_weight+' '+m.unit+' | '+m.captured_at;result.appendChild(details);
   if(m.image_url){const image=document.createElement('img');image.src=m.image_url;image.alt='Ảnh lần cân';result.appendChild(image);}
  }}catch(error){result.innerHTML='<div class="error">Lỗi tra cứu: '+error.message+'</div>';}
 input.value='';input.focus();});
</script></body></html>"""


def serve(api_url: str, token: str, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                self._send(200, "text/html; charset=utf-8", LOOKUP_HTML.encode("utf-8"))
                return
            if parsed.path != "/api/lookup":
                self._send(404, "application/json", b'{"ok":false,"error":"not_found"}')
                return
            qr_code = urllib.parse.parse_qs(parsed.query).get("qr", [""])[0].strip()
            if not qr_code:
                self._send(422, "application/json", b'{"ok":false,"error":"missing_qr"}')
                return
            try:
                result = lookup_roll(api_url, qr_code, token)
                body = json.dumps(result, ensure_ascii=False).encode("utf-8")
                self._send(200 if result.get("found") else 404, "application/json", body)
            except Exception as exc:
                body = json.dumps(
                    {"ok": False, "error": "lookup_failed", "message": str(exc)},
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send(502, "application/json", body)

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Lookup UI: http://{host}:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def run(args: argparse.Namespace) -> int:
    if not args.api_url or not args.api_token:
        raise ValueError("Lookup requires --api-url and --api-token")
    if args.serve:
        serve(args.api_url, args.api_token, args.host, args.port)
        return 0

    qr_code = args.qr or (_lookup_from_image(args.image) if args.image else None)
    if qr_code:
        result = lookup_roll(args.api_url, qr_code, args.api_token)
        print(_format_result(result))
        return 0 if result.get("found") else 2

    print("Scanner HID mode. Scan a QR and press Ctrl+C to stop.")
    while True:
        qr_code = input("QR> ").strip()
        if not qr_code:
            continue
        print(_format_result(lookup_roll(args.api_url, qr_code, args.api_token)))


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(run(args))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
