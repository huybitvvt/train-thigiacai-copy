-- Store the Cloudinary identifiers returned for each captured measurement.
-- image_path remains populated for compatibility with older Supabase Storage records.
alter table public.measurements
  add column if not exists image_url text,
  add column if not exists image_public_id text;

create index if not exists measurements_image_public_id_idx
  on public.measurements (image_public_id)
  where image_public_id is not null;
