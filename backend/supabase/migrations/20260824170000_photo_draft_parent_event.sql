-- Gắn từng ảnh chụp không-AI vào đúng event và ô của phiếu cân.

alter table public.anh_can_cho_ai
  add column if not exists parent_event_id uuid,
  add column if not exists capture_kind text not null default 'core',
  add column if not exists capture_round integer not null default 0;

update public.anh_can_cho_ai
set parent_event_id = event_id
where parent_event_id is null;

alter table public.anh_can_cho_ai
  alter column parent_event_id set not null;

alter table public.anh_can_cho_ai
  drop constraint if exists anh_can_cho_ai_capture_kind_check,
  drop constraint if exists anh_can_cho_ai_capture_round_check;

alter table public.anh_can_cho_ai
  add constraint anh_can_cho_ai_capture_kind_check
    check (capture_kind in ('core', 'product')),
  add constraint anh_can_cho_ai_capture_round_check
    check (capture_round between 0 and 3);

create index if not exists anh_can_cho_ai_parent_event_idx
  on public.anh_can_cho_ai (parent_event_id, capture_round, capture_kind);

comment on column public.anh_can_cho_ai.parent_event_id is
  'Event của phiếu cân chứa ảnh; nhiều ảnh/ô có thể dùng chung event này.';
comment on column public.anh_can_cho_ai.capture_kind is
  'Ô ảnh trong phiếu cân: core hoặc product.';
comment on column public.anh_can_cho_ai.capture_round is
  'Số thứ tự lượt cân, bắt đầu từ 0.';
