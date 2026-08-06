$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dest = Join-Path $root "dist\source-handoff-0.2.0-rc2-clean"
$zip = Join-Path $root "dist\source-handoff-0.2.0-rc2-clean.zip"
$hashFile = Join-Path $root "dist\source-handoff-0.2.0-rc2-clean.SHA256.txt"

if (Test-Path -LiteralPath $dest) {
    throw "Đích đã tồn tại, không ghi đè: $dest"
}
if (Test-Path -LiteralPath $zip) {
    throw "ZIP đã tồn tại, không ghi đè: $zip"
}
New-Item -ItemType Directory -Path $dest -Force | Out-Null

function Copy-Tree([string]$relative) {
    $source = Join-Path $root $relative
    Get-ChildItem -LiteralPath $source -Recurse -File -Force |
        Where-Object {
            $_.FullName -notmatch '\\(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.temp)(\\|$)' -and
            $_.FullName -notmatch '\\[^\\]+\.egg-info(\\|$)' -and
            $_.Extension -notin @(".pyc", ".pyo")
        } |
        ForEach-Object {
            $subpath = $_.FullName.Substring($root.Length + 1)
            $target = Join-Path $dest $subpath
            New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $target -Force
        }
}

foreach ($directory in @("src", "tests", "tools", "packaging", "docs", "supabase", "config")) {
    Copy-Tree $directory
}

foreach ($file in @(
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    ".env.example",
    "supabase_schema.sql",
    "YOLOv8_QR_Can_Colab.ipynb"
)) {
    Copy-Item -LiteralPath (Join-Path $root $file) -Destination (Join-Path $dest $file) -Force
}

foreach ($file in @(
    "data\test_frame.png",
    "data\warehouse_scale_demo.png",
    "data\warehouse_scale_demo_base.png",
    "data\factory_scale_7_02_full_reference.jpg",
    "data\factory_scale_7_02_full_reference.json",
    "data\factory_scale_9_34_reference.jpg",
    "data\captures\20260802_175736_868757_65389c95.jpg",
    "models\qr_demo_synthetic.pt",
    "yolov8n.pt"
)) {
    $source = Join-Path $root $file
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Thiếu file source cần bàn giao: $file"
    }
    $target = Join-Path $dest $file
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $target -Force
}

$entries = Get-ChildItem -LiteralPath $dest -Force | Select-Object -ExpandProperty FullName
Compress-Archive -Path $entries -DestinationPath $zip -CompressionLevel Optimal
$hash = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash
$hash | Set-Content -LiteralPath $hashFile -Encoding ascii

[pscustomobject]@{
    Folder = $dest
    Zip = $zip
    ZipBytes = (Get-Item -LiteralPath $zip).Length
    Sha256 = $hash
    FileCount = (Get-ChildItem -LiteralPath $dest -Recurse -File -Force | Measure-Object).Count
    HasEnvExample = Test-Path -LiteralPath (Join-Path $dest ".env.example")
    HasDatabase = [bool](Get-ChildItem -LiteralPath $dest -Recurse -File -Force -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -in @(".db", ".db-wal", ".db-shm") })
    HasRealConfig = [bool](Get-ChildItem -LiteralPath $dest -Recurse -File -Filter "config.env" -ErrorAction SilentlyContinue)
} | Format-List
