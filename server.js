// server.js  (ESM)
// Requires: npm i express cors multer openai
// Node 18+ (Render uses Node 22) so global fetch is available.

import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI from "openai";

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

/* ============================================================
   Request logging + request id (Render-friendly)
   ============================================================ */
function mkReqId() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

app.use((req, res, next) => {
  req.reqId = mkReqId();
  const start = Date.now();

  console.log(
    `[${req.reqId}] --> ${req.method} ${req.originalUrl} ua="${(req.headers["user-agent"] || "")
      .toString()
      .slice(0, 120)}"`
  );

  res.on("finish", () => {
    const ms = Date.now() - start;
    console.log(`[${req.reqId}] <-- ${req.method} ${req.originalUrl} ${res.statusCode} ${ms}ms`);
  });

  // real abort only (avoid false positives from "close")
  req.on("aborted", () => {
    console.warn(`[${req.reqId}] !! request aborted by client`);
  });

  next();
});

/* ============================================================
   Uploads: in-memory (Render-friendly)
   ============================================================ */
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB
});

/* ============================================================
   OpenAI
   ============================================================ */
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.get("/health", (_, res) => res.json({ ok: true, ts: Date.now() }));

/* ============================================================
   Debug auth (optional)
   If DEBUG_TOKEN is set on Render, debug endpoints require header:
   x-debug-token: <DEBUG_TOKEN>
   ============================================================ */
const DEBUG_TOKEN = process.env.DEBUG_TOKEN || "";

function requireDebugAuth(req, res, next) {
  if (!DEBUG_TOKEN) return next(); // open if not configured
  const got = (req.headers["x-debug-token"] || "").toString();
  if (got !== DEBUG_TOKEN) return res.status(401).json({ error: "Unauthorized debug token" });
  next();
}

/* ============================================================
   GEO config (slow-but-concrete: used in async job)
   ============================================================ */
const OVERPASS_TIMEOUT_MS = Number(process.env.OVERPASS_TIMEOUT_MS || 20000);
const NOMINATIM_TIMEOUT_MS = Number(process.env.NOMINATIM_TIMEOUT_MS || 10000);
const OVERPASS_RADIUS_M = Number(process.env.OVERPASS_RADIUS_M || 1200);
const POI_TARGET_COUNT = Number(process.env.POI_TARGET_COUNT || 8);

const OSM_CONTACT = process.env.OSM_CONTACT_EMAIL || "dev@yourdomain.com";
const OSM_UA = `travel-vision-backend/1.0 (contact: ${OSM_CONTACT})`;

const OVERPASS_ENDPOINTS = [
  "https://overpass-api.de/api/interpreter",
  "https://overpass.kumi.systems/api/interpreter",
  "https://overpass.openstreetmap.ru/api/interpreter"
];

/* ============================================================
   Helpers
   ============================================================ */
async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(t);
  }
}

function haversineMeters(lat1, lon1, lat2, lon2) {
  const R = 6371000;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

function buildOverpassQuery(lat, lon, radiusM) {
  // Attraction-ish POIs with names
  return `
[out:json][timeout:25];
(
  nwr(around:${radiusM},${lat},${lon})["tourism"]["name"];
  nwr(around:${radiusM},${lat},${lon})["historic"]["name"];
  nwr(around:${radiusM},${lat},${lon})["amenity"="place_of_worship"]["name"];
  nwr(around:${radiusM},${lat},${lon})["building"="temple"]["name"];
  nwr(around:${radiusM},${lat},${lon})["man_made"]["name"];
  nwr(around:${radiusM},${lat},${lon})["leisure"="park"]["name"];
);
out center 80;
`.trim();
}

async function fetchOverpassWithFallback(query, reqId, timeoutOverrideMs) {
  let lastErr = null;
  const timeoutMs = Number.isFinite(timeoutOverrideMs) ? timeoutOverrideMs : OVERPASS_TIMEOUT_MS;

  for (const endpoint of OVERPASS_ENDPOINTS) {
    const t0 = Date.now();
    try {
      console.log(`[${reqId}] Overpass try ${endpoint} timeout=${timeoutMs}ms`);
      const resp = await fetchWithTimeout(
        endpoint,
        {
          method: "POST",
          headers: {
            "Content-Type": "text/plain",
            "User-Agent": OSM_UA
          },
          body: query
        },
        timeoutMs
      );
      const ms = Date.now() - t0;

      if (!resp.ok) {
        const txt = await resp.text().catch(() => "");
        throw new Error(`Overpass HTTP ${resp.status} (${ms}ms) body=${txt.slice(0, 140)}`);
      }

      const json = await resp.json();
      console.log(`[${reqId}] Overpass OK ${endpoint} (${ms}ms) elements=${json?.elements?.length || 0}`);
      return json;
    } catch (e) {
      lastErr = e;
      console.warn(`[${reqId}] Overpass fail ${endpoint}: ${e?.message || e}`);
    }
  }

  throw lastErr || new Error("Overpass failed (no endpoints succeeded)");
}

async function fetchNearbyPois(lat, lon, radiusM, reqId, timeoutOverrideMs) {
  const overpassQuery = buildOverpassQuery(lat, lon, radiusM);
  const json = await fetchOverpassWithFallback(overpassQuery, reqId, timeoutOverrideMs);
  const elements = Array.isArray(json?.elements) ? json.elements : [];

  const poiList = elements
    .map((el) => {
      const name = el?.tags?.name;
      if (!name) return null;

      const pLat = el.lat ?? el.center?.lat;
      const pLon = el.lon ?? el.center?.lon;
      if (!Number.isFinite(pLat) || !Number.isFinite(pLon)) return null;

      const tags = el.tags || {};
      const type =
        tags.tourism ||
        tags.historic ||
        (tags.amenity === "place_of_worship" ? "place_of_worship" : null) ||
        (tags.building === "temple" ? "temple" : null) ||
        tags.man_made ||
        (tags.leisure === "park" ? "park" : null) ||
        "poi";

      const distance_m = Math.round(haversineMeters(lat, lon, pLat, pLon));

      const hint = [
        tags.tourism && `tourism=${tags.tourism}`,
        tags.historic && `historic=${tags.historic}`,
        tags.amenity && `amenity=${tags.amenity}`,
        tags.building && `building=${tags.building}`,
        tags.man_made && `man_made=${tags.man_made}`,
        tags.leisure && `leisure=${tags.leisure}`
      ]
        .filter(Boolean)
        .slice(0, 3)
        .join(", ");

      return { name: String(name), type: String(type), distance_m, hint };
    })
    .filter(Boolean);

  // distance sort, de-dupe by name, keep up to 25
  const seen = new Set();
  const out = [];
  for (const p of poiList.sort((a, b) => a.distance_m - b.distance_m)) {
    const key = p.name.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(p);
    if (out.length >= 25) break;
  }

  return out;
}

async function reverseGeocode(lat, lon, reqId) {
  // Primary: Nominatim
  const url = `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${encodeURIComponent(
    lat
  )}&lon=${encodeURIComponent(lon)}&zoom=18&addressdetails=1`;

  const t0 = Date.now();
  try {
    const resp = await fetchWithTimeout(
      url,
      {
        headers: {
          "User-Agent": OSM_UA,
          "Accept-Language": "en",
          "Referer": "https://travel-vision-backend.local/"
        }
      },
      NOMINATIM_TIMEOUT_MS
    );
    const ms = Date.now() - t0;

    if (!resp.ok) throw new Error(`Nominatim reverse failed: ${resp.status} (${ms}ms)`);
    const json = await resp.json();
    const addr = json?.address || {};
    const out = {
      displayName: json?.display_name || null,
      city: addr.city || addr.town || addr.village || addr.municipality || addr.county || null,
      state: addr.state || null,
      country: addr.country || null
    };
    console.log(`[${reqId}] Nominatim OK (${ms}ms) "${(out.displayName || "").slice(0, 80)}"`);
    return out;
  } catch (e) {
    console.warn(`[${reqId}] Nominatim failed, fallback: ${e?.message || e}`);

    // Fallback: BigDataCloud (no key)
    const fb = `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${encodeURIComponent(
      lat
    )}&longitude=${encodeURIComponent(lon)}&localityLanguage=en`;

    const resp2 = await fetchWithTimeout(fb, { headers: { "User-Agent": OSM_UA } }, 8000);
    if (!resp2.ok) throw new Error(`Fallback reverse failed: ${resp2.status}`);
    const j = await resp2.json();

    const displayName =
      [j?.locality, j?.principalSubdivision, j?.countryName].filter(Boolean).join(", ") || null;

    const out = {
      displayName,
      city: j?.city || j?.locality || null,
      state: j?.principalSubdivision || null,
      country: j?.countryName || null
    };
    console.log(`[${reqId}] Fallback reverse OK "${(out.displayName || "").slice(0, 80)}"`);
    return out;
  }
}

/* ============================================================
   Locate job system (prevents client/proxy timeouts)
   ============================================================ */
const JOB_TTL_MS = Number(process.env.JOB_TTL_MS || 15 * 60 * 1000);
const locateJobs = new Map(); // jobId -> job

function makeJobId() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

function trace(job, step, detail) {
  const item = { ts: Date.now(), step, detail };
  job.trace.push(item);
  console.log(`[job:${job.jobId}] ${step} ${detail}`);
}

setInterval(() => {
  const now = Date.now();
  for (const [id, job] of locateJobs.entries()) {
    if (now - job.createdAt > JOB_TTL_MS) locateJobs.delete(id);
  }
}, 60_000);

async function runLocateJob(job) {
  try {
    job.status = "geo";
    trace(job, "geo_start", `lat=${job.lat} lon=${job.lon} baseR=${OVERPASS_RADIUS_M}m`);

    // 1) Reverse first
    const reverse = await reverseGeocode(job.lat, job.lon, `job:${job.jobId}`);
    job.geo.reverse = reverse;
    trace(job, "reverse_done", `${(reverse.displayName || "").slice(0, 120)}`);

    // 2) Overpass next (increase radius until we get enough)
    const radii = [
      OVERPASS_RADIUS_M,
      Math.round(OVERPASS_RADIUS_M * 2),
      Math.round(OVERPASS_RADIUS_M * 4)
    ];

    let pois = [];
    let usedRadius = radii[0];

    for (const r of radii) {
      usedRadius = r;
      trace(job, "overpass_start", `radius=${r}m timeout=${OVERPASS_TIMEOUT_MS}ms`);
      const got = await fetchNearbyPois(job.lat, job.lon, r, `job:${job.jobId}`);
      trace(job, "overpass_done", `radius=${r}m pois=${got.length}`);
      pois = got;
      if (pois.length >= POI_TARGET_COUNT) break;
    }

    job.geo.radiusM = usedRadius;
    job.geo.pois = pois;

    // 3) OpenAI last
    job.status = "openai";
    trace(job, "openai_start", `pois=${pois.length} reverse="${(reverse.displayName || "").slice(0, 60)}"`);

    const hasPois = pois.length > 0;

    const locationBlock = `Location is available and HIGH PRIORITY.
Coordinates: lat=${job.lat}, lon=${job.lon}${Number.isFinite(job.acc) ? ` (accuracy ≈ ${job.acc}m)` : ""}.
Reverse-geocoded area: ${reverse.displayName || "(unavailable)"}.

Nearby places from map data (within ${usedRadius}m):
${
  hasPois
    ? pois
        .slice(0, 20)
        .map((p) => `- ${p.name} (${p.type}, ~${p.distance_m}m)${p.hint ? ` [${p.hint}]` : ""}`)
        .join("\n")
    : "(none returned)"
}

CRITICAL RULES:
- Use real named places close to the provided coordinates as a helping tool.
- Use the nearby-places list as strong evidence, but you may choose outside it if the photo suggests something else.`;

    const schema = {
      type: "object",
      properties: {
        photoContext: { type: "string" },
        candidates: {
          type: "array",
          minItems: 3,
          maxItems: 3,
          items: {
            type: "object",
            properties: {
              name: { type: "string" },
              why: { type: "string" },
              confidence: { type: "number", minimum: 0, maximum: 1 },
              searchQuery: { type: "string" }
            },
            required: ["name", "why", "confidence", "searchQuery"],
            additionalProperties: false
          }
        }
      },
      required: ["photoContext", "candidates"],
      additionalProperties: false
    };

    const prompt = `
You are a travel expert.

Task A — PHOTO CONTEXT (3–4 sentences):
Describe what is visible in THIS photo (viewpoint, lighting/time, weather if visible, materials, architecture, signs, crowd level, surroundings, writing).
Do NOT name/guess the place in this section.

Task B — IDENTIFY PLACE:
Return EXACTLY 3 candidates ranked best -> worst.
For each candidate:
- name: short friendly place name with city/country if possible
- why: 1–2 sentences referencing visual cues AND how coordinates + nearby list affected the choice
- confidence: 0..1
- searchQuery: likely query to find a representative image

${locationBlock}

User request text:
${job.userText ? job.userText : "(none)"}

Rules:
- Don't invent certainty, if image unrecognisable tell so.
- If photo contradicts coords, mention it in "why" and lower confidence.
`.trim();

    job.debug.prompt = prompt;

    const response = await client.responses.create({
      model: "gpt-4.1-mini",
      temperature: 0,
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: prompt },
            { type: "input_image", image_url: job.dataUrl }
          ]
        }
      ],
      text: {
        format: {
          type: "json_schema",
          name: "locate_with_context",
          strict: true,
          schema
        }
      }
    });

    const raw = response.output_text;
    job.debug.openaiRaw = raw.slice(0, 4000);

    const parsed = JSON.parse(raw);
    if (!parsed?.candidates || parsed.candidates.length !== 3) {
      throw new Error("Model did not return 3 candidates");
    }

    job.result = {
      photoContext: parsed.photoContext,
      candidates: parsed.candidates
    };

    job.status = "done";
    trace(job, "openai_done", parsed.candidates.map((c) => c.name).join(" | "));
  } catch (e) {
    job.status = "error";
    job.error = String(e?.message || e);
    trace(job, "error", job.error);
  } finally {
    // free big blob
    job.dataUrl = null;
  }
}

/* ============================================================
   /api/locate  -> returns { jobId } quickly
   Requires lat/lon (this build is location-first)
   ============================================================ */
app.post("/api/locate", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "Missing form field 'image'." });

    const userText = (req.body?.message || "").toString().trim();
    const lat = req.body?.lat != null ? Number(req.body.lat) : null;
    const lon = req.body?.lon != null ? Number(req.body.lon) : null;
    const acc = req.body?.accuracyMeters != null ? Number(req.body.accuracyMeters) : null;

    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      return res.status(400).json({ error: "Missing/invalid lat/lon. This server requires location." });
    }

    const mime = req.file.mimetype || "image/jpeg";
    const base64 = req.file.buffer.toString("base64");
    const dataUrl = `data:${mime};base64,${base64}`;

    const jobId = makeJobId();
    const job = {
      jobId,
      createdAt: Date.now(),
      status: "pending",
      error: null,
      trace: [],
      userText,
      lat,
      lon,
      acc,
      dataUrl,
      geo: { reverse: null, pois: [], radiusM: null },
      result: null,
      debug: { prompt: null, openaiRaw: null }
    };

    locateJobs.set(jobId, job);

    console.log(
      `[${req.reqId}] LOCATE job created jobId=${jobId} lat=${lat} lon=${lon} acc=${acc} bytes=${req.file.buffer.length}`
    );

    setImmediate(() => runLocateJob(job));

    return res.json({ jobId });
  } catch (e) {
    console.error(`[${req.reqId}] /api/locate ERROR`, e);
    return res.status(500).json({ error: "Server error" });
  }
});

/* ============================================================
   /api/locate_result?jobId=...
   -> pending/geo/openai/done/error
   ============================================================ */
app.get("/api/locate_result", async (req, res) => {
  const jobId = (req.query.jobId || "").toString();
  const job = locateJobs.get(jobId);
  if (!job) return res.status(404).json({ error: "Unknown jobId" });

  if (job.status === "pending" || job.status === "geo" || job.status === "openai") {
    return res.json({
      status: job.status,
      reverse: job.geo?.reverse?.displayName || null,
      poisCount: job.geo?.pois?.length || 0
    });
  }

  if (job.status === "error") {
    return res.json({ status: "error", error: job.error || "Unknown error" });
  }

  return res.json({
    status: "done",
    photoContext: job.result?.photoContext || "",
    candidates: job.result?.candidates || []
  });
});

/* ============================================================
   /api/chat
   - Android sends photoContext (no server caching)
   - Android can send factsInstruction (user-customizable)
   ============================================================ */
app.post("/api/chat", async (req, res) => {
  try {
    const place = (req.body?.place || "").toString().trim();
    const message = (req.body?.message || "").toString().trim();
    const photoContext = (req.body?.photoContext || "").toString().trim();

    // NEW: user-defined facts instructions (from Android settings page)
    const factsInstruction = (req.body?.factsInstruction || "").toString().trim();

    if (!place) return res.status(400).json({ error: "Missing JSON field 'place'." });

    const isFacts = !message || /^facts?$|^guide$|^info$|^history$/i.test(message);

    const defaultFactsInstruction = `
- 1 short opener referencing the photo context if available
- 1–2 lines: what it is + why it’s famous
- 3 bullet historical highlights (if unsure say "unknown")
- 3 bullet fun facts
- 3 bullet practical visiting tips relevant to the photo context
Keep it short and readable.
`.trim();

    const effectiveFactsInstruction = factsInstruction || defaultFactsInstruction;

    const prompt = isFacts
      ? `
You are a friendly travel guide.

Place: ${place}
Photo context (from user's last uploaded photo): ${photoContext || "(not available)"}

Follow THESE user preferences exactly:
${effectiveFactsInstruction}

Rules:
- Don't invent specifics. If uncertain, say "unknown".
- Keep it helpful and practical.
`.trim()
      : `
You are a friendly travel guide.

Place: ${place}
Photo context: ${photoContext || "(not available)"}
User question: ${message}

Answer clearly and practically. If photo context helps, reference it briefly.
If user asks for prices/hours/tickets "today", say you may be out of date and suggest checking official sources.
Keep it readable and not too long. Mention Ivan briefly in some context.
`.trim();

    const t0 = Date.now();
    const response = await client.responses.create({
      model: "gpt-4.1-mini",
      temperature: 0,
      input: [{ role: "user", content: prompt }]
    });
    console.log(`[${req.reqId}] CHAT OpenAI ${Date.now() - t0}ms`);

    return res.json({ text: response.output_text });
  } catch (e) {
    console.error(`[${req.reqId}] /api/chat ERROR`, e);
    return res.status(500).json({ error: "Server error" });
  }
});

/* ============================================================
   Debug endpoints (use from Android Debug screen)
   ============================================================ */

// Test Overpass from Render (may be slow; you can pass &timeoutMs=1200 to cap)
app.get("/debug/overpass", requireDebugAuth, async (req, res) => {
  try {
    const lat = Number(req.query.lat);
    const lon = Number(req.query.lon);
    const r = Number(req.query.r || OVERPASS_RADIUS_M);
    const timeoutMs = req.query.timeoutMs != null ? Number(req.query.timeoutMs) : undefined;

    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      return res.status(400).json({ ok: false, error: "Missing lat/lon" });
    }

    const q = buildOverpassQuery(lat, lon, r);
    const t0 = Date.now();
    const json = await fetchOverpassWithFallback(q, `debug:${req.reqId}`, timeoutMs);
    const ms = Date.now() - t0;

    const elements = Array.isArray(json?.elements) ? json.elements : [];
    const names = elements
      .map((e) => e?.tags?.name)
      .filter(Boolean)
      .slice(0, 20);

    res.json({ ok: true, ms, radiusM: r, elements: elements.length, sampleNames: names });
  } catch (e) {
    res.status(200).json({ ok: false, error: String(e?.message || e) });
  }
});

// Test reverse geocode (Nominatim with fallback)
app.get("/debug/reverse", requireDebugAuth, async (req, res) => {
  try {
    const lat = Number(req.query.lat);
    const lon = Number(req.query.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      return res.status(400).json({ ok: false, error: "Missing lat/lon" });
    }
    const out = await reverseGeocode(lat, lon, `debug:${req.reqId}`);
    res.json({ ok: true, reverse: out });
  } catch (e) {
    res.status(200).json({ ok: false, error: String(e?.message || e) });
  }
});

// Test OpenAI text (no image)
app.post("/debug/openai_text", requireDebugAuth, async (req, res) => {
  try {
    const prompt = (req.body?.prompt || "").toString();
    if (!prompt) return res.status(400).json({ ok: false, error: "Missing prompt" });

    const t0 = Date.now();
    const r = await client.responses.create({
      model: "gpt-4.1-mini",
      temperature: 0,
      input: [{ role: "user", content: prompt }]
    });
    res.json({ ok: true, ms: Date.now() - t0, text: r.output_text });
  } catch (e) {
    res.status(200).json({ ok: false, error: String(e?.message || e) });
  }
});

// Inspect job (trace + prompt preview + openai raw preview)
app.get("/debug/job", requireDebugAuth, async (req, res) => {
  const jobId = (req.query.jobId || "").toString();
  const job = locateJobs.get(jobId);
  if (!job) return res.status(404).json({ ok: false, error: "Unknown jobId" });

  res.json({
    ok: true,
    jobId,
    status: job.status,
    error: job.error || null,
    createdAt: job.createdAt,
    lat: job.lat,
    lon: job.lon,
    acc: job.acc,
    geo: {
      reverse: job.geo.reverse,
      radiusM: job.geo.radiusM,
      poisCount: job.geo.pois.length,
      samplePois: job.geo.pois.slice(0, 10)
    },
    trace: job.trace.slice(-200),
    debug: {
      promptPreview: (job.debug.prompt || "").slice(0, 2000),
      openaiRawPreview: (job.debug.openaiRaw || "").slice(0, 2000)
    }
  });
});

/* ============================================================
   Start
   ============================================================ */
const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Server running on port ${port}`));
