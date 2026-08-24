from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FUNCTION = (
    ROOT / "backend" / "supabase" / "functions" / "ingest-measurement" / "index.ts"
).read_text(encoding="utf-8")
MIGRATION = (
    ROOT
    / "backend"
    / "supabase"
    / "migrations"
    / "20260824150000_anh_can_cho_ai.sql"
).read_text(encoding="utf-8")


def test_photo_draft_table_keeps_weight_data_truly_empty() -> None:
    table_sql = MIGRATION.lower().split("create table", 1)[1].split(");", 1)[0]
    assert "public.anh_can_cho_ai" in table_sql
    assert "qr_code text" in table_sql
    assert "status text not null default 'awaiting_ai'" in table_sql
    assert "weight" not in table_sql
    assert "khoi_luong" not in table_sql


def test_ingest_routes_photo_draft_before_measurement_validation() -> None:
    assert 'const photoDraft = workflow === "photo_draft"' in FUNCTION
    assert 'const PHOTO_DRAFT_TABLE = "anh_can_cho_ai"' in FUNCTION
    assert 'if (!photoDraft && (!Number.isFinite(weight)' in FUNCTION
    assert 'status: "awaiting_ai"' in FUNCTION
    assert 'ai_requested: false' in FUNCTION
    assert "/photo-draft/${eventId}" in FUNCTION
