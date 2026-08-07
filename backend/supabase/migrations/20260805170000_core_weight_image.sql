-- Keep the scale-core evidence image separate from the product image owned by
-- the production application. The legacy image_* columns remain populated for
-- clients deployed before this distinction existed.
alter table public.can_tu_dong
  add column if not exists core_image_path text,
  add column if not exists core_image_url text,
  add column if not exists core_image_public_id text;

update public.can_tu_dong
set
  core_image_path = coalesce(core_image_path, image_path),
  core_image_url = coalesce(core_image_url, image_url),
  core_image_public_id = coalesce(core_image_public_id, image_public_id)
where core_image_path is null
   or core_image_url is null
   or core_image_public_id is null;

comment on column public.can_tu_dong.core_image_path is
  'Cloudinary public ID/path for the image captured while reading core weight.';
comment on column public.can_tu_dong.core_image_url is
  'Cloudinary delivery URL for the core-weight evidence image (ẢNH TL LÕI).';
comment on column public.can_tu_dong.core_image_public_id is
  'Cloudinary public ID for the core-weight evidence image.';

create index if not exists can_tu_dong_core_image_public_id_idx
  on public.can_tu_dong (core_image_public_id)
  where core_image_public_id is not null;

-- The legacy table is retained for installations that still query it.
alter table public.measurements
  add column if not exists core_image_path text,
  add column if not exists core_image_url text,
  add column if not exists core_image_public_id text;

update public.measurements
set
  core_image_path = coalesce(core_image_path, image_path),
  core_image_url = coalesce(core_image_url, image_url),
  core_image_public_id = coalesce(core_image_public_id, image_public_id)
where core_image_path is null
   or core_image_url is null
   or core_image_public_id is null;
