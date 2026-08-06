import base64
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import qrcode


NOTEBOOK = Path(__file__).parents[1] / "YOLOv8_QR_Can_Colab.ipynb"


class FakeResponse:
    data = [{"id": 42}]


class FakeTable:
    def __init__(self, state: dict):
        self.state = state

    def insert(self, row: dict):
        self.state["row"] = row
        return self

    def execute(self):
        return FakeResponse()


class FakeBucket:
    def __init__(self, state: dict):
        self.state = state

    def upload(self, **kwargs):
        self.state["upload"] = kwargs

    def remove(self, paths):
        self.state["removed"] = paths


class FakeStorage:
    def __init__(self, state: dict):
        self.state = state

    def from_(self, name: str):
        self.state["bucket"] = name
        return FakeBucket(self.state)


class FakeSupabase:
    def __init__(self):
        self.state: dict = {}
        self.storage = FakeStorage(self.state)

    def table(self, name: str):
        self.state["table"] = name
        return FakeTable(self.state)


def _cell_source(prefix: str) -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "".join(cell["source"]).startswith(prefix)
    )


def test_notebook_is_valid_json_and_code_cells_compile() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(line for line in cell["source"] if not line.lstrip().startswith(("%", "!")))
        if source.strip():
            compile(source, f"cell-{index}", "exec")


def test_colab_callback_decodes_and_inserts_supabase(monkeypatch) -> None:
    registered: dict[str, object] = {}
    output_stub = types.SimpleNamespace(
        register_callback=lambda name, function: registered.setdefault(name, function)
    )
    colab_stub = types.ModuleType("google.colab")
    colab_stub.output = output_stub
    monkeypatch.setitem(sys.modules, "google.colab", colab_stub)

    class FakeJSON:
        def __init__(self, data: dict):
            self.data = data

    ipython_stub = types.ModuleType("IPython")
    display_stub = types.ModuleType("IPython.display")
    display_stub.JSON = FakeJSON
    ipython_stub.display = display_stub
    monkeypatch.setitem(sys.modules, "IPython", ipython_stub)
    monkeypatch.setitem(sys.modules, "IPython.display", display_stub)

    fake = FakeSupabase()
    namespace = {
        "np": np,
        "supabase": fake,
        "BUCKET_NAME": "roll-captures",
    }
    exec(_cell_source("# Backend:"), namespace)

    qr = qrcode.make("ROLL-SUPABASE-001").convert("RGB").resize((500, 500))
    frame = cv2.cvtColor(np.asarray(qr), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    data_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")

    result = namespace["capture_callback"](data_url, "125,4", "kg")
    assert result.data["ok"] is True
    assert result.data["id"] == 42
    assert fake.state["bucket"] == "roll-captures"
    assert fake.state["table"] == "measurements"
    assert fake.state["row"]["qr_code"] == "ROLL-SUPABASE-001"
    assert fake.state["row"]["weight"] == 125.4
