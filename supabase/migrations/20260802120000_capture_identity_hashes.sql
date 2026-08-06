-- Add durable multi-camera identity and integrity metadata without rewriting
-- historical rows. NULL means the record predates that field.
alter table public.measurements
  add column if not exists gateway_id text,
  add column if not exists station_id text,
  add column if not exists camera_id text,
  add column if not exists analysis_id text,
  add column if not exists frame_sha256 text,
  add column if not exists payload_hash text;

alter table public.can_tu_dong
  add column if not exists gateway_id text,
  add column if not exists station_id text,
  add column if not exists camera_id text,
  add column if not exists analysis_id text,
  add column if not exists frame_sha256 text,
  add column if not exists payload_hash text;

do $$
begin
  if not exists (
    select 1 from pg_constraint
    where conrelid = 'public.measurements'::regclass
      and conname = 'measurements_frame_sha256_check'
  ) then
    alter table public.measurements
      add constraint measurements_frame_sha256_check
      check (frame_sha256 is null or frame_sha256 ~* '^[0-9a-f]{64}$');
  end if;
  if not exists (
    select 1 from pg_constraint
    where conrelid = 'public.measurements'::regclass
      and conname = 'measurements_payload_hash_check'
  ) then
    alter table public.measurements
      add constraint measurements_payload_hash_check
      check (payload_hash is null or payload_hash ~* '^[0-9a-f]{64}$');
  end if;
  if not exists (
    select 1 from pg_constraint
    where conrelid = 'public.can_tu_dong'::regclass
      and conname = 'can_tu_dong_frame_sha256_check'
  ) then
    alter table public.can_tu_dong
      add constraint can_tu_dong_frame_sha256_check
      check (frame_sha256 is null or frame_sha256 ~* '^[0-9a-f]{64}$');
  end if;
  if not exists (
    select 1 from pg_constraint
    where conrelid = 'public.can_tu_dong'::regclass
      and conname = 'can_tu_dong_payload_hash_check'
  ) then
    alter table public.can_tu_dong
      add constraint can_tu_dong_payload_hash_check
      check (payload_hash is null or payload_hash ~* '^[0-9a-f]{64}$');
  end if;
end $$;

create index if not exists measurements_capture_identity_idx
  on public.measurements (gateway_id, station_id, camera_id, captured_at desc);
create index if not exists measurements_analysis_id_idx
  on public.measurements (analysis_id)
  where analysis_id is not null;

create index if not exists can_tu_dong_capture_identity_idx
  on public.can_tu_dong (gateway_id, station_id, camera_id, captured_at desc);
create index if not exists can_tu_dong_analysis_id_idx
  on public.can_tu_dong (analysis_id)
  where analysis_id is not null;

comment on column public.can_tu_dong.gateway_id is
  'Stable gateway/computer identity. device_id remains populated for compatibility.';
comment on column public.can_tu_dong.analysis_id is
  'Opaque ID for the local frame-analysis attempt that produced this event.';
comment on column public.can_tu_dong.frame_sha256 is
  'Lowercase SHA-256 of the exact JPEG bytes uploaded to Cloudinary.';
comment on column public.can_tu_dong.payload_hash is
  'Lowercase SHA-256 of the canonical immutable measurement payload.';
