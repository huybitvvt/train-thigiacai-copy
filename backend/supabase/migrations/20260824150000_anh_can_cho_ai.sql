-- Ảnh cân chờ AI: một ảnh được lưu ngay, chưa bắt buộc mã hoặc số cân.

create table if not exists public.anh_can_cho_ai (
  id bigserial primary key,
  event_id uuid not null unique,
  qr_code text,
  captured_at timestamptz not null,
  image_path text not null,
  image_url text not null,
  image_public_id text not null,
  gateway_id text not null,
  station_id text,
  camera_id text,
  frame_sha256 text,
  payload_hash text,
  qr_source text not null default 'none',
  work_date date,
  shift text,
  machine text,
  production_order text,
  status text not null default 'awaiting_ai'
    check (status in ('awaiting_ai', 'processing', 'processed', 'failed')),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists anh_can_cho_ai_status_time_idx
  on public.anh_can_cho_ai (status, captured_at);
create index if not exists anh_can_cho_ai_qr_idx
  on public.anh_can_cho_ai (qr_code)
  where qr_code is not null;

comment on table public.anh_can_cho_ai is
  'Ảnh cân lưu độc lập để AI đọc nối tiếp sau; số cân được để trống.';

alter table public.anh_can_cho_ai enable row level security;
revoke all on table public.anh_can_cho_ai from anon, authenticated;
