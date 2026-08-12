-- Encrypted ChatGPT/Codex OAuth payloads for trusted scale gateways.
-- Only the service role used inside the ingest Edge Function may access them.
create table if not exists public.roll_scale_secrets (
    name text primary key,
    encrypted_value text not null,
    updated_at timestamptz not null default now(),
    constraint roll_scale_secrets_name_format
        check (name ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$'),
    constraint roll_scale_secrets_value_length
        check (length(encrypted_value) between 1 and 16384)
);

alter table public.roll_scale_secrets enable row level security;
revoke all on table public.roll_scale_secrets from anon, authenticated;
grant all on table public.roll_scale_secrets to service_role;
