import { createClient } from "npm:@supabase/supabase-js@2";

const MAX_IMAGE_BYTES = 6 * 1024 * 1024;
const UNITS = new Set(["kg", "g", "lb"]);
const MEASUREMENT_TABLE = "can_tu_dong";
const SECRET_TABLE = "roll_scale_secrets";
const ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/;
const SHA256_PATTERN = /^[0-9a-f]{64}$/i;
const EVENT_SELECT =
  "id,image_path,image_url,image_public_id,core_image_path,core_image_url," +
  "core_image_public_id,product_image_path,product_image_url,product_image_public_id,qr_code,weight,tare_weight,net_weight,unit,captured_at," +
  "device_id,gateway_id,station_id,camera_id,analysis_id,frame_sha256,payload_hash," +
  "weight_source,qr_source,metadata";

function json(status: number, body: Record<string, unknown>): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

function decodeBase64(value: string): Uint8Array {
  const binary = atob(value);
  return Uint8Array.from(binary, (character) => character.charCodeAt(0));
}

async function sha256Hex(value: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", Uint8Array.from(value));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function getSupabaseAdminKey(): string | undefined {
  const secretKeys = Deno.env.get("SUPABASE_SECRET_KEYS");
  if (secretKeys) {
    try {
      const parsed = JSON.parse(secretKeys) as Record<string, unknown>;
      if (typeof parsed.default === "string") return parsed.default;
    } catch {
      // Fall through to the legacy key used by older Supabase projects.
    }
  }
  return Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
}

function sameEvent(
  existing: Record<string, unknown>,
  qrCode: string,
  weight: number,
  unit: string,
  capturedAt: string,
  weightSource: string,
  qrSource: string,
  weightRaw: string,
  weightStable: boolean,
  gatewayId: string,
  stationId: string | null,
  cameraId: string | null,
  analysisId: string | null,
  frameSha256: string | null,
  payloadHash: string | null,
): boolean {
  const metadata = existing.metadata !== null && typeof existing.metadata === "object"
    ? existing.metadata as Record<string, unknown>
    : {};
  const existingGateway = existing.gateway_id ?? existing.device_id;
  const storedTare = Number(existing.tare_weight ?? 0);
  const existingCoreWeight = Number(
    metadata.core_weight ?? (storedTare > 0 ? storedTare : existing.weight),
  );
  return existing.qr_code === qrCode &&
    existingCoreWeight === weight &&
    existing.unit === unit &&
    Date.parse(String(existing.captured_at)) === Date.parse(capturedAt) &&
    existing.weight_source === weightSource &&
    existing.qr_source === qrSource &&
    storedValueMatches(metadata.weight_raw, weightRaw) &&
    storedValueMatches(metadata.weight_stable, weightStable) &&
    storedValueMatches(existingGateway, gatewayId) &&
    storedValueMatches(existing.station_id, stationId) &&
    storedValueMatches(existing.camera_id, cameraId) &&
    storedValueMatches(existing.analysis_id, analysisId) &&
    storedHashMatches(existing.frame_sha256, frameSha256) &&
    storedHashMatches(existing.payload_hash, payloadHash);
}

// A NULL stored value identifies a row created before that field existed. Such
// rows accept the richer retry payload. Once stored, an immutable field must be
// supplied and match on every retry of the same event_id.
function storedValueMatches(existing: unknown, incoming: unknown): boolean {
  if (existing === null || existing === undefined) return true;
  return incoming !== null && incoming !== undefined && existing === incoming;
}

function storedHashMatches(existing: unknown, incoming: string | null): boolean {
  if (existing === null || existing === undefined) return true;
  return incoming !== null && String(existing).toLowerCase() === incoming;
}

type CloudinaryUpload = {
  publicId: string;
  secureUrl: string;
};

async function uploadToCloudinary(
  image: Uint8Array,
  publicId: string,
): Promise<CloudinaryUpload> {
  const cloudName = Deno.env.get("CLOUDINARY_CLOUD_NAME");
  const apiKey = Deno.env.get("CLOUDINARY_API_KEY");
  const apiSecret = Deno.env.get("CLOUDINARY_API_SECRET");
  if (!cloudName || !apiKey || !apiSecret) {
    throw new Error("cloudinary_not_configured");
  }

  const form = new FormData();
  form.append(
    "file",
    new Blob([Uint8Array.from(image)], { type: "image/jpeg" }),
    `${publicId.split("/").at(-1)}.jpg`,
  );
  form.append("public_id", publicId);
  form.append("overwrite", "false");

  const response = await fetch(
    `https://api.cloudinary.com/v1_1/${encodeURIComponent(cloudName)}/image/upload`,
    {
      method: "POST",
      headers: { authorization: `Basic ${btoa(`${apiKey}:${apiSecret}`)}` },
      body: form,
    },
  );
  let result: Record<string, unknown> = {};
  try {
    result = await response.json() as Record<string, unknown>;
  } catch {
    // The status below remains the authoritative failure signal.
  }
  if (!response.ok) {
    throw new Error(`cloudinary_upload_failed:${response.status}`);
  }
  const secureUrl = typeof result.secure_url === "string" ? result.secure_url : "";
  const returnedPublicId = typeof result.public_id === "string" ? result.public_id : "";
  if (!secureUrl || !returnedPublicId) {
    throw new Error("cloudinary_invalid_response");
  }
  return { publicId: returnedPublicId, secureUrl };
}

Deno.serve(async (request: Request) => {
  if (request.method !== "POST" && request.method !== "GET") {
    return json(405, { ok: false, error: "method_not_allowed" });
  }

  const expectedToken = Deno.env.get("DEVICE_INGEST_TOKEN");
  const suppliedToken = request.headers.get("x-device-token");
  if (!expectedToken) {
    return json(500, { ok: false, error: "server_not_configured" });
  }
  if (!suppliedToken || suppliedToken !== expectedToken) {
    return json(401, { ok: false, error: "unauthorized" });
  }

  const requestUrl = new URL(request.url);
  const action = requestUrl.searchParams.get("action") ?? "";

  if (request.method === "GET" && (action === "codex-auth" || action === "encrypted-secret")) {
    const name = requestUrl.searchParams.get("name")?.trim() ?? "";
    if (!ID_PATTERN.test(name)) {
      return json(422, { ok: false, error: "invalid_secret_name" });
    }
    const supabaseUrl = Deno.env.get("SUPABASE_URL");
    const serviceKey = getSupabaseAdminKey();
    if (!supabaseUrl || !serviceKey) {
      return json(500, { ok: false, error: "supabase_not_configured" });
    }
    const supabase = createClient(supabaseUrl, serviceKey, {
      auth: { persistSession: false, autoRefreshToken: false },
    });
    const { data, error } = await supabase
      .from(SECRET_TABLE)
      .select("encrypted_value")
      .eq("name", name)
      .maybeSingle();
    if (error) {
      return json(500, { ok: false, error: "secret_read_failed" });
    }
    return json(200, {
      ok: true,
      found: Boolean(data),
      encrypted_value: data?.encrypted_value ?? null,
    });
  }

  if (request.method === "GET") {
    const supabaseUrl = Deno.env.get("SUPABASE_URL");
    const serviceKey = getSupabaseAdminKey();
    if (!supabaseUrl || !serviceKey) {
      return json(500, { ok: false, error: "supabase_not_configured" });
    }
    const requestedLimit = Number(new URL(request.url).searchParams.get("limit") ?? "50");
    const limit = Number.isInteger(requestedLimit)
      ? Math.max(1, Math.min(requestedLimit, 200))
      : 50;
    const supabase = createClient(supabaseUrl, serviceKey, {
      auth: { persistSession: false, autoRefreshToken: false },
    });
    const { data, error } = await supabase
      .from(MEASUREMENT_TABLE)
      .select(EVENT_SELECT)
      .order("captured_at", { ascending: false })
      .limit(limit);
    if (error) {
      return json(500, { ok: false, error: "measurement_list_failed" });
    }
    return json(200, { ok: true, source: MEASUREMENT_TABLE, items: data ?? [] });
  }

  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return json(400, { ok: false, error: "invalid_json" });
  }

  if (body.action === "codex-auth" || body.action === "encrypted-secret") {
    const name = typeof body.name === "string" ? body.name.trim() : "";
    const encryptedValue = typeof body.encrypted_value === "string"
      ? body.encrypted_value.trim()
      : "";
    if (!ID_PATTERN.test(name)) {
      return json(422, { ok: false, error: "invalid_secret_name" });
    }
    if (!encryptedValue || encryptedValue.length > 16384) {
      return json(422, { ok: false, error: "invalid_encrypted_value" });
    }
    const supabaseUrl = Deno.env.get("SUPABASE_URL");
    const serviceKey = getSupabaseAdminKey();
    if (!supabaseUrl || !serviceKey) {
      return json(500, { ok: false, error: "supabase_not_configured" });
    }
    const supabase = createClient(supabaseUrl, serviceKey, {
      auth: { persistSession: false, autoRefreshToken: false },
    });
    const { error } = await supabase.from(SECRET_TABLE).upsert(
      {
        name,
        encrypted_value: encryptedValue,
        updated_at: new Date().toISOString(),
      },
      { onConflict: "name" },
    );
    if (error) {
      return json(500, { ok: false, error: "secret_write_failed" });
    }
    return json(200, { ok: true, stored: true });
  }

  const eventId = typeof body.event_id === "string" ? body.event_id : "";
  const qrCode = typeof body.qr_code === "string" ? body.qr_code.trim() : "";
  const suppliedGatewayId = typeof body.gateway_id === "string" ? body.gateway_id.trim() : null;
  const legacyDeviceId = typeof body.device_id === "string" ? body.device_id.trim() : null;
  const gatewayId = suppliedGatewayId || legacyDeviceId || "";
  const stationId = typeof body.station_id === "string" ? body.station_id.trim() : null;
  const cameraId = typeof body.camera_id === "string" ? body.camera_id.trim() : null;
  const analysisId = typeof body.analysis_id === "string" ? body.analysis_id.trim() : null;
  const frameSha256 = typeof body.frame_sha256 === "string"
    ? body.frame_sha256.trim().toLowerCase()
    : null;
  const payloadHash = typeof body.payload_hash === "string"
    ? body.payload_hash.trim().toLowerCase()
    : null;
  const unit = typeof body.unit === "string" ? body.unit : "";
  const weight = typeof body.weight === "number" ? body.weight : Number.NaN;
  const productWeight = typeof body.product_weight === "number"
    ? body.product_weight
    : Number.NaN;
  const capturedAt = typeof body.captured_at === "string" ? body.captured_at : "";
  const imageBase64 = typeof body.image_base64 === "string" ? body.image_base64 : "";
  const productImageBase64 = typeof body.product_image_base64 === "string"
    ? body.product_image_base64
    : "";
  const imageRole = typeof body.image_role === "string" ? body.image_role : "";
  const weightSource = typeof body.weight_source === "string"
    ? body.weight_source.slice(0, 100)
    : "unknown";
  const qrSource = typeof body.qr_source === "string" ? body.qr_source.slice(0, 100) : "unknown";
  const weightRaw = typeof body.weight_raw === "string" ? body.weight_raw.slice(0, 1000) : "";
  const weightStable = body.weight_stable === true;
  const sourceTag = (name: string): string => {
    const match = weightRaw.match(new RegExp(`(?:^|; )\\s*${name}=([^;]+)`));
    return match ? match[1].trim().slice(0, 80) : "";
  };
  const workDate = sourceTag("SOURCE_DATE");
  const shift = sourceTag("SOURCE_SHIFT");
  const machine = sourceTag("SOURCE_MACHINE");
  const productionOrder = sourceTag("SOURCE_PRODUCTION_ORDER");
  const biWeightRaw = sourceTag("BI_WEIGHT");
  const biWeightParsed = Number(biWeightRaw);
  const biWeight = Number.isFinite(biWeightParsed) && biWeightParsed >= 0
    ? biWeightParsed
    : 0.16;

  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(eventId)) {
    return json(422, { ok: false, error: "invalid_event_id" });
  }
  if (!qrCode || qrCode.length > 512) {
    return json(422, { ok: false, error: "invalid_qr_code" });
  }
  if (body.gateway_id !== undefined && body.gateway_id !== null && !suppliedGatewayId) {
    return json(422, { ok: false, error: "invalid_gateway_id" });
  }
  if (body.device_id !== undefined && body.device_id !== null && !legacyDeviceId) {
    return json(422, { ok: false, error: "invalid_device_id" });
  }
  if (!ID_PATTERN.test(gatewayId)) {
    return json(422, { ok: false, error: "invalid_gateway_id" });
  }
  if (suppliedGatewayId && legacyDeviceId && suppliedGatewayId !== legacyDeviceId) {
    return json(422, { ok: false, error: "gateway_device_id_mismatch" });
  }
  for (
    const [field, value] of [
      ["station_id", stationId],
      ["camera_id", cameraId],
      ["analysis_id", analysisId],
    ] as const
  ) {
    if (body[field] !== undefined && body[field] !== null && (!value || !ID_PATTERN.test(value))) {
      return json(422, { ok: false, error: `invalid_${field}` });
    }
  }
  if (
    body.frame_sha256 !== undefined && body.frame_sha256 !== null &&
    (!frameSha256 || !SHA256_PATTERN.test(frameSha256))
  ) {
    return json(422, { ok: false, error: "invalid_frame_sha256" });
  }
  if (
    body.payload_hash !== undefined && body.payload_hash !== null &&
    (!payloadHash || !SHA256_PATTERN.test(payloadHash))
  ) {
    return json(422, { ok: false, error: "invalid_payload_hash" });
  }
  if (!Number.isFinite(weight) || weight < 0 || !UNITS.has(unit)) {
    return json(422, { ok: false, error: "invalid_weight" });
  }
  if (!capturedAt || Number.isNaN(Date.parse(capturedAt))) {
    return json(422, { ok: false, error: "invalid_captured_at" });
  }
  if (imageBase64.length > Math.ceil(MAX_IMAGE_BYTES / 3) * 4 + 4) {
    return json(413, { ok: false, error: "image_too_large" });
  }
  if (!Number.isFinite(productWeight) || productWeight < 0) {
    return json(422, { ok: false, error: "invalid_product_weight" });
  }
  if (productImageBase64.length > Math.ceil(MAX_IMAGE_BYTES / 3) * 4 + 4) {
    return json(413, { ok: false, error: "product_image_too_large" });
  }
  if (imageRole && imageRole !== "core_weight") {
    return json(422, { ok: false, error: "invalid_image_role" });
  }

  let image: Uint8Array;
  try {
    image = decodeBase64(imageBase64);
  } catch {
    return json(422, { ok: false, error: "invalid_image_base64" });
  }
  if (
    image.length < 4 || image.length > MAX_IMAGE_BYTES ||
    image[0] !== 0xff || image[1] !== 0xd8
  ) {
    return json(422, { ok: false, error: "invalid_jpeg" });
  }
  if (frameSha256 && await sha256Hex(image) !== frameSha256) {
    return json(422, { ok: false, error: "frame_sha256_mismatch" });
  }
  let productImage: Uint8Array | null = null;
  if (productImageBase64) {
    try {
      productImage = decodeBase64(productImageBase64);
    } catch {
      return json(422, { ok: false, error: "invalid_product_image_base64" });
    }
    if (
      productImage.length < 4 || productImage.length > MAX_IMAGE_BYTES ||
      productImage[0] !== 0xff || productImage[1] !== 0xd8
    ) {
      return json(422, { ok: false, error: "invalid_product_jpeg" });
    }
  }

  const supabaseUrl = Deno.env.get("SUPABASE_URL");
  const serviceKey = getSupabaseAdminKey();
  if (!supabaseUrl || !serviceKey) {
    return json(500, { ok: false, error: "supabase_not_configured" });
  }
  const supabase = createClient(supabaseUrl, serviceKey, {
    auth: { persistSession: false, autoRefreshToken: false },
  });

  const { data: existing, error: lookupError } = await supabase
    .from(MEASUREMENT_TABLE)
    .select(EVENT_SELECT)
    .eq("event_id", eventId)
    .maybeSingle();
  if (lookupError) {
    return json(500, { ok: false, error: "lookup_failed" });
  }
  if (existing) {
    const existingRow = existing as unknown as Record<string, unknown>;
    if (
      !sameEvent(
        existingRow,
        qrCode,
        weight,
        unit,
        capturedAt,
        weightSource,
        qrSource,
        weightRaw,
        weightStable,
        gatewayId,
        stationId,
        cameraId,
        analysisId,
        frameSha256,
        payloadHash,
      )
    ) {
      return json(409, { ok: false, error: "event_id_conflict" });
    }
    const existingProductUrl = typeof existingRow.product_image_url === "string"
      ? existingRow.product_image_url
      : "";
    const productIsOnCloudinary = existingProductUrl.startsWith(
      "https://res.cloudinary.com/",
    );
    if (productImage && !productIsOnCloudinary) {
      const captureDate = new Date(capturedAt);
      const year = captureDate.getUTCFullYear();
      const month = String(captureDate.getUTCMonth() + 1).padStart(2, "0");
      const day = String(captureDate.getUTCDate()).padStart(2, "0");
      let productUploaded: CloudinaryUpload;
      try {
        productUploaded = await uploadToCloudinary(
          productImage,
          `roll-captures/${gatewayId}/${year}/${month}/${day}/product-weight/${eventId}`,
        );
      } catch (error) {
        const message = error instanceof Error ? error.message : "cloudinary_upload_failed";
        return json(500, { ok: false, error: message.split(":", 1)[0] });
      }
      const existingMetadata = existingRow.metadata !== null &&
          typeof existingRow.metadata === "object"
        ? existingRow.metadata as Record<string, unknown>
        : {};
      const { data: updated, error: updateError } = await supabase
        .from(MEASUREMENT_TABLE)
        .update({
          product_image_path: productUploaded.publicId,
          product_image_url: productUploaded.secureUrl,
          product_image_public_id: productUploaded.publicId,
          metadata: { ...existingMetadata, product_weight: productWeight },
        })
        .eq("event_id", eventId)
        .select(EVENT_SELECT)
        .single();
      if (updateError || !updated) {
        return json(500, { ok: false, error: "product_image_update_failed" });
      }
      const updatedRow = updated as unknown as Record<string, unknown>;
      return json(200, {
        ok: true,
        id: updatedRow.id,
        event_id: eventId,
        image_url: updatedRow.image_url,
        image_public_id: updatedRow.image_public_id,
        core_image_url: updatedRow.core_image_url ?? updatedRow.image_url,
        core_image_public_id: updatedRow.core_image_public_id ?? updatedRow.image_public_id,
        product_image_url: updatedRow.product_image_url,
        product_image_public_id: updatedRow.product_image_public_id,
        duplicate: true,
      });
    }
    return json(200, {
      ok: true,
      id: existingRow.id,
      event_id: eventId,
      image_url: existingRow.image_url,
      image_public_id: existingRow.image_public_id,
      core_image_url: existingRow.core_image_url ?? existingRow.image_url,
      core_image_public_id: existingRow.core_image_public_id ?? existingRow.image_public_id,
      product_image_url: existingRow.product_image_url,
      product_image_public_id: existingRow.product_image_public_id,
      gateway_id: existingRow.gateway_id ?? existingRow.device_id,
      station_id: existingRow.station_id,
      camera_id: existingRow.camera_id,
      analysis_id: existingRow.analysis_id,
      frame_sha256: existingRow.frame_sha256,
      payload_hash: existingRow.payload_hash,
      duplicate: true,
    });
  }

  const now = new Date().toISOString();
  const { error: deviceError } = await supabase.from("devices").upsert(
    { id: gatewayId, last_seen_at: now },
    { onConflict: "id" },
  );
  if (deviceError) {
    return json(500, { ok: false, error: "device_upsert_failed" });
  }
  const { error: rollError } = await supabase.from("rolls").upsert(
    { qr_code: qrCode, last_seen_at: now },
    { onConflict: "qr_code" },
  );
  if (rollError) {
    return json(500, { ok: false, error: "roll_upsert_failed" });
  }

  const captureDate = new Date(capturedAt);
  const year = captureDate.getUTCFullYear();
  const month = String(captureDate.getUTCMonth() + 1).padStart(2, "0");
  const day = String(captureDate.getUTCDate()).padStart(2, "0");
  const imagePublicId = `roll-captures/${gatewayId}/${year}/${month}/${day}/core-weight/${eventId}`;
  let uploaded: CloudinaryUpload;
  let productUploaded: CloudinaryUpload | null = null;
  try {
    uploaded = await uploadToCloudinary(image, imagePublicId);
    if (productImage) {
      productUploaded = await uploadToCloudinary(
        productImage,
        `roll-captures/${gatewayId}/${year}/${month}/${day}/product-weight/${eventId}`,
      );
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "cloudinary_upload_failed";
    return json(500, { ok: false, error: message.split(":", 1)[0] });
  }

  const { data: inserted, error: insertError } = await supabase
    .from(MEASUREMENT_TABLE)
    .insert({
      event_id: eventId,
      qr_code: qrCode,
      // In can_tu_dong, weight is the gross/product reading. The core reading
      // is the tare, so the generated net_weight is gross minus core.
      weight: productWeight,
      tare_weight: weight,
      unit,
      captured_at: capturedAt,
      image_path: uploaded.publicId,
      image_url: uploaded.secureUrl,
      image_public_id: uploaded.publicId,
      core_image_path: uploaded.publicId,
      core_image_url: uploaded.secureUrl,
      core_image_public_id: uploaded.publicId,
      product_image_path: productUploaded?.publicId ?? null,
      product_image_url: productUploaded?.secureUrl ?? null,
      product_image_public_id: productUploaded?.publicId ?? null,
      device_id: gatewayId,
      gateway_id: gatewayId,
      station_id: stationId,
      camera_id: cameraId,
      analysis_id: analysisId,
      frame_sha256: frameSha256,
      payload_hash: payloadHash,
      weight_source: weightSource,
      qr_source: qrSource,
      status: "confirmed",
      metadata: {
        ingested_at: now,
        weight_raw: weightRaw,
        weight_stable: weightStable,
        core_weight: weight,
        product_weight: productWeight,
        work_date: workDate || null,
        shift: shift || null,
        machine: machine || null,
        production_order: productionOrder || null,
        bi_weight: biWeight,
      },
    })
    .select("id,image_path")
    .single();

  if (insertError) {
    if (insertError.code === "23505") {
      const { data: raced } = await supabase
        .from(MEASUREMENT_TABLE)
        .select(EVENT_SELECT)
        .eq("event_id", eventId)
        .single();
      if (raced) {
        const racedRow = raced as unknown as Record<string, unknown>;
        if (
          !sameEvent(
            racedRow,
            qrCode,
            weight,
            unit,
            capturedAt,
            weightSource,
            qrSource,
            weightRaw,
            weightStable,
            gatewayId,
            stationId,
            cameraId,
            analysisId,
            frameSha256,
            payloadHash,
          )
        ) {
          return json(409, { ok: false, error: "event_id_conflict" });
        }
        return json(200, {
          ok: true,
          id: racedRow.id,
          event_id: eventId,
          image_url: racedRow.image_url,
          image_public_id: racedRow.image_public_id,
          core_image_url: racedRow.core_image_url ?? racedRow.image_url,
          core_image_public_id: racedRow.core_image_public_id ?? racedRow.image_public_id,
          product_image_url: racedRow.product_image_url,
          product_image_public_id: racedRow.product_image_public_id,
          gateway_id: racedRow.gateway_id ?? racedRow.device_id,
          station_id: racedRow.station_id,
          camera_id: racedRow.camera_id,
          analysis_id: racedRow.analysis_id,
          frame_sha256: racedRow.frame_sha256,
          payload_hash: racedRow.payload_hash,
          duplicate: true,
        });
      }
    }
    return json(500, { ok: false, error: "measurement_insert_failed" });
  }

  return json(201, {
    ok: true,
    id: inserted.id,
    event_id: eventId,
    image_url: uploaded.secureUrl,
    image_public_id: uploaded.publicId,
    core_image_url: uploaded.secureUrl,
    core_image_public_id: uploaded.publicId,
    product_image_url: productUploaded?.secureUrl ?? null,
    product_image_public_id: productUploaded?.publicId ?? null,
    gateway_id: gatewayId,
    station_id: stationId,
    camera_id: cameraId,
    analysis_id: analysisId,
    frame_sha256: frameSha256,
    payload_hash: payloadHash,
    duplicate: false,
  });
});
