-- Store the second evidence image captured specifically for the product QR.
-- Existing rows remain valid and keep these fields NULL.
alter table public.can_tu_dong
  add column if not exists qr_image_path text,
  add column if not exists qr_image_url text,
  add column if not exists qr_image_public_id text,
  add column if not exists qr_frame_sha256 text;

alter table public.measurements
  add column if not exists qr_image_path text,
  add column if not exists qr_image_url text,
  add column if not exists qr_image_public_id text,
  add column if not exists qr_frame_sha256 text;

do $$
begin
  if not exists (
    select 1 from pg_constraint
    where conrelid = 'public.can_tu_dong'::regclass
      and conname = 'can_tu_dong_qr_frame_sha256_check'
  ) then
    alter table public.can_tu_dong
      add constraint can_tu_dong_qr_frame_sha256_check
      check (qr_frame_sha256 is null or qr_frame_sha256 ~* '^[0-9a-f]{64}$');
  end if;
  if not exists (
    select 1 from pg_constraint
    where conrelid = 'public.measurements'::regclass
      and conname = 'measurements_qr_frame_sha256_check'
  ) then
    alter table public.measurements
      add constraint measurements_qr_frame_sha256_check
      check (qr_frame_sha256 is null or qr_frame_sha256 ~* '^[0-9a-f]{64}$');
  end if;
end $$;

comment on column public.can_tu_dong.qr_image_url is
  'Cloudinary delivery URL for the second image captured to decode the product QR.';
comment on column public.can_tu_dong.qr_frame_sha256 is
  'Lowercase SHA-256 of the exact second QR-evidence JPEG uploaded to Cloudinary.';

create index if not exists can_tu_dong_qr_image_public_id_idx
  on public.can_tu_dong (qr_image_public_id)
  where qr_image_public_id is not null;
