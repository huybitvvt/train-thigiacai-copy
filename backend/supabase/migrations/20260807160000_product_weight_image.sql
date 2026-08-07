alter table if exists public.can_tu_dong
  add column if not exists product_weight double precision,
  add column if not exists product_image_path text,
  add column if not exists product_image_url text,
  add column if not exists product_image_public_id text;

alter table if exists public.measurements
  add column if not exists product_weight double precision,
  add column if not exists product_image_path text,
  add column if not exists product_image_url text,
  add column if not exists product_image_public_id text;

comment on column public.can_tu_dong.product_image_url is
  'Cloudinary URL of the product-weight evidence captured after the core weight.';
