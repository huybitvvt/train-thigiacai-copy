alter table if exists public.can_tu_dong
  add column if not exists product_weight double precision,
  add column if not exists product_image_path text,
  add column if not exists product_image_url text,
  add column if not exists product_image_public_id text;

grant select on table public.can_tu_dong to anon;

alter table public.can_tu_dong enable row level security;

do $$
begin
  if not exists (
    select 1
    from pg_policies
    where schemaname = 'public'
      and tablename = 'can_tu_dong'
      and policyname = 'anon_read_can_tu_dong'
  ) then
    create policy anon_read_can_tu_dong
      on public.can_tu_dong
      for select
      to anon
      using (true);
  end if;
end
$$;
