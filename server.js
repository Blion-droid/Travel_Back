import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI from "openai";

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

/* ============================================================
   Basic request logging (Render-friendly: stdout/stderr)
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
      .slice(0, 80)}"`
  );

  res.on("finish", () => {
    const ms = Date.now() - start;
    console.log(
      `[${req.reqId}] <-- ${req.method} ${req.originalUrl} ${res.statusCode} ${ms}ms`
    );
  });

  next();
});

// In-memory uploads (Render-friendly)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB
});

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.get("/health", (_, res) => res.json({ ok: true, ts: Date.now() }));

/* ============================================================
   OPTION A: Server-side photoContext cache (NO app changes)
   ============================================================ */

const PHOTO_TTL_MS = 10 * 60 * 1000; // 10 minutes
const photoContextCache = new Map();

/** best-effort key: IP + User-Agent reduces collisions */
function cacheKeyFromReq(req) {
  const xf = req.headers["x-forwarded-for"];
  const ip =
    (typeof xf === "string" ? xf.split(",")[0].trim() : null) ||
    req.ip ||
    "unknown";
  const ua = (req.headers["user-agent"] || "ua:unknown").toString().slice(0, 200);
  return `${ip}__${ua}`;
}

function setPhotoContext(req, photoContext) {
  const key = cacheKeyFromReq(req);
  photoContextCache.set(key, { photoContext, expiresAt: Date.now() + PHOTO_TTL_MS });
}

function getPhotoContext(req) {
  const key = cacheKeyFromReq(req);
  const entry = photoContextCache.get(key);
  if (!entry) return null;
  if (Date.now() > entry.expiresAt) {
    photoContextCache.delete(key);
    return null;
  }
  return entry.photoContext;
}

// Cleanup once/min to avoid memory growth
setInterval(() => {
  const now = Date.now();
  for (const [k, v] of photoContextCache.entries()) {
    if (now > v.expiresAt) photoContextCache.delete(k);
  }
}, 60_000);

/* ============================================================
   GEO HELPERS (OSM Nominatim + Overpass) + caching + logging
   Works worldwide as far as OSM coverage allows.
   ============================================================ */

const GEO_TTL_MS = 10 * 60 * 1000; // 10 minutes
const geoCache = new Map();

// IMPORTANT: put a real contact email for OSM services.
// You can set this via env var on Render to avoid hardcoding.
const OSM_CONTACT = process.env.OSM_CONTACT_EMAIL || "you@example.com";
const OSM_UA = `travel-locate-server/1.0 (contact: ${OSM_CONTACT})`;

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

function geoCacheKey(lat, lon, radiusM) {
  const rLat = Math.round(lat * 1e5) / 1e5;
  const rLon = Math.round(lon * 1e5) / 1e5;
  return `${rLat},${rLon},r=${radiusM}`;
}

async function fetchWithTimeout(url, options, timeoutMs = 8000) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, { ...options, signal: controller.signal });
    return resp;
  } finally {
    clearTimeout(t);
  }
}

async function reverseGeocode(lat, lon) {
  const url =
    `https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat=${encodeURIComponent(
      lat
    )}&lon=${encodeURIComponent(lon)}&zoom=18&addressdetails=1`;

  const resp = await fetchWithTimeout(
    url,
    {
      headers: {
        "User-Agent": OSM_UA,
        "Accept-Language": "en"
      }
    },
    8000
  );

  if (!resp.ok) throw new Error(`Nominatim reverse failed: ${resp.status}`);
  const json = await resp.json();

  const addr = json?.address || {};
  return {
    displayName: json?.display_name || null,
    city: addr.city || addr.town || addr.village || addr.municipality || addr.county || null,
    state: addr.state || null,
    country: addr.country || null
  };
}

/**
 * Attraction-focused Overpass query.
 * nwr = node/way/relation.
 */
function buildOverpassQuery(lat, lon, radiusM) {
  return `
[out:json][timeout:20];
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

async function fetchNearbyPois(lat, lon, radiusM) {
  const overpassQuery = buildOverpassQuery(lat, lon, radiusM);

  const resp = await fetchWithTimeout(
    "https://overpass-api.de/api/interpreter",
    {
      method: "POST",
      headers: {
        "Content-Type": "text/plain",
        "User-Agent": OSM_UA
      },
      body: overpassQuery
    },
    10000
  );

  if (!resp.ok) throw new Error(`Overpass failed: ${resp.status}`);
  const json = await resp.json();

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

      return {
        name: String(name),
        type: String(type),
        distance_m,
        hint
      };
    })
    .filter(Boolean);

  // Sort by distance, de-dupe by name, keep small for prompt size
  const seen = new Set();
  const deduped = [];
  for (const p of poiList.sort((a, b) => a.distance_m - b.distance_m)) {
    const key = p.name.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(p);
    if (deduped.length >= 25) break;
  }

  return deduped;
}

/**
 * Dynamic radius strategy:
 * - Start 1500m
 * - If POIs too few (<6), expand to 3000m
 * - If still too few (<6), expand to 5000m
 */
async function getGeoContext(lat, lon, reqId) {
  const radii = [1500, 3000, 5000];

  for (const radiusM of radii) {
    const key = geoCacheKey(lat, lon, radiusM);
    const cached = geoCache.get(key);
    if (cached && Date.now() < cached.expiresAt) {
      console.log(`[${reqId}] GEO cache hit radius=${radiusM}m pois=${cached.value.pois.length}`);
      return cached.value;
    }

    console.log(`[${reqId}] GEO lookup start radius=${radiusM}m`);
    const t0 = Date.now();

    const [rev, pois] = await Promise.all([
      reverseGeocode(lat, lon).catch((e) => {
        console.warn(`[${reqId}] Nominatim failed: ${e?.message || e}`);
        return { displayName: null, city: null, state: null, country: null };
      }),
      fetchNearbyPois(lat, lon, radiusM)
    ]);

    const ms = Date.now() - t0;
    console.log(
      `[${reqId}] GEO lookup done radius=${radiusM}m pois=${pois.length} ms=${ms} reverse="${(
        rev.displayName || ""
      ).slice(0, 90)}"`
    );

    const value = { reverse: rev, pois, radiusM };
    geoCache.set(key, { value, expiresAt: Date.now() + GEO_TTL_MS });

    if (pois.length >= 6 || radiusM === radii[radii.length - 1]) {
      return value;
    }

    console.log(`[${reqId}] GEO too few POIs (${pois.length}); expanding radius...`);
  }

  return { reverse: { displayName: null }, pois: [], radiusM: 1500 };
}

// Cleanup geo cache once/min
setInterval(() => {
  const now = Date.now();
  for (const [k, v] of geoCache.entries()) {
    if (now > v.expiresAt) geoCache.delete(k);
  }
}, 60_000);

/* ============================================================
   /api/locate
   - Generates photoContext + candidates
   - Caches photoContext server-side
   - Returns ONLY candidates (so Android doesn't need changes)
   - Uses reverse-geocode + nearby POIs to avoid “wild” guesses
   ============================================================ */
app.post("/api/locate", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "Missing form field 'image'." });

    const userText = (req.body?.message || "").toString().trim();
    const lat = req.body?.lat != null ? Number(req.body.lat) : null;
    const lon = req.body?.lon != null ? Number(req.body.lon) : null;
    const acc = req.body?.accuracyMeters != null ? Number(req.body.accuracyMeters) : null;

    const mime = req.file.mimetype || "image/jpeg";
    const base64 = req.file.buffer.toString("base64");
    const dataUrl = `data:${mime};base64,${base64}`;

    const hasLoc = Number.isFinite(lat) && Number.isFinite(lon);

    console.log(
      `[${req.reqId}] LOCATE hasLoc=${hasLoc} lat=${lat} lon=${lon} acc=${acc} userTextLen=${userText.length}`
    );

    // NEW: geo context lookup (reverse + POIs) for better grounding
    let geo = null;
    if (hasLoc) {
      try {
        geo = await getGeoContext(lat, lon, req.reqId);
      } catch (err) {
        console.warn(
          `[${req.reqId}] Geo context lookup failed; proceeding without POIs: ${err?.message || err}`
        );
        geo = null;
      }
    }

    const locationBlock = hasLoc
      ? `Location is available and HIGH PRIORITY.
Coordinates: lat=${lat}, lon=${lon}${Number.isFinite(acc) ? ` (accuracy ≈ ${acc}m)` : ""}.
Reverse-geocoded area (may be approximate): ${geo?.reverse?.displayName || "(unavailable)"}.

Nearby places from map data (within ${geo?.radiusM || 1500}m):
${
  geo?.pois?.length
    ? geo.pois
        .slice(0, 20)
        .map((p) => `- ${p.name} (${p.type}, ~${p.distance_m}m)${p.hint ? ` [${p.hint}]` : ""}`)
        .join("\n")
    : "(none returned)"
}

CRITICAL RULES:
- If a nearby-places list is provided and is not empty, you MUST choose the 3 candidates FROM THAT LIST.
- Only choose something outside the list if the PHOTO CLEARLY contradicts the coordinates; if so, explicitly include the phrase "coordinate conflict" in why AND set confidence <= 0.2.`
      : `Location is NOT available. Rely on visual cues + user text.`;

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
- name: short friendly place name WITH city/country if possible
- why: 1–2 sentences referencing visual cues AND (if provided) how coordinates affected the choice. If distances are provided, say "very close" (hundreds of meters) or "farther" (over ~1km).
- confidence: 0..1
- searchQuery: likely query to find a representative image

${locationBlock}

User request text:
${userText ? userText : "(none)"}

Rules:
- Don't invent certainty.
- If photo contradicts coords, mention it in "why" and lower confidence.
- If a nearby-places list is provided and non-empty, pick ONLY from that list unless you explicitly flag "coordinate conflict" (confidence <= 0.2).
`.trim();

    console.log(
      `[${req.reqId}] Prompt built. hasPOIs=${Boolean(geo?.pois?.length)} poisCount=${geo?.pois?.length || 0}`
    );

    const tModel0 = Date.now();
    const response = await client.responses.create({
      model: "gpt-4.1-mini",
      temperature: 0,
      input: [
        {
          role: "user",
          content: [
            { type: "input_text", text: prompt },
            { type: "input_image", image_url: dataUrl }
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
    console.log(`[${req.reqId}] Model responded in ${Date.now() - tModel0}ms`);

    const parsed = JSON.parse(response.output_text);

    if (parsed?.photoContext) setPhotoContext(req, parsed.photoContext);

    // OPTIONAL: If we have POIs and the model ignored them without "coordinate conflict", do one repair retry
    if (hasLoc && geo?.pois?.length) {
      const allowed = new Set(geo.pois.map((p) => p.name.toLowerCase()));
      const chosen = (parsed?.candidates || []).map((c) => (c?.name || "").toLowerCase());

      const anyMatches = chosen.some((nm) => {
        for (const a of allowed) {
          if (nm.includes(a) || a.includes(nm)) return true;
        }
        return false;
      });

      const mentionsConflict = (parsed?.candidates || []).some((c) =>
        String(c?.why || "").toLowerCase().includes("coordinate conflict")
      );

      if (!anyMatches && !mentionsConflict) {
        console.warn(
          `[${req.reqId}] Model candidates did not match POI list; attempting 1 repair retry. chosen=${JSON.stringify(
            parsed?.candidates || []
          ).slice(0, 300)}`
        );

        const repairPrompt = `
You must pick EXACTLY 3 candidates FROM THIS LIST (do not invent others).
If the photo clearly contradicts the GPS, you may pick outside the list but then:
- include the phrase "coordinate conflict" in why
- set confidence <= 0.2

Nearby places list:
${geo.pois
  .slice(0, 20)
  .map((p) => `- ${p.name} (${p.type}, ~${p.distance_m}m)`)
  .join("\n")}

Return JSON with the same schema: { photoContext, candidates[3] }.
`.trim();

        const tRepair0 = Date.now();
        const repairResp = await client.responses.create({
          model: "gpt-4.1-mini",
          temperature: 0,
          input: [
            {
              role: "user",
              content: [
                { type: "input_text", text: repairPrompt },
                { type: "input_image", image_url: dataUrl }
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
        console.log(`[${req.reqId}] Repair model responded in ${Date.now() - tRepair0}ms`);

        const repaired = JSON.parse(repairResp.output_text);
        if (repaired?.photoContext) setPhotoContext(req, repaired.photoContext);

        if (repaired?.candidates?.length === 3) {
          console.log(
            `[${req.reqId}] Returning repaired candidates: ${repaired.candidates
              .map((c) => c.name)
              .join(" | ")}`
          );
          return res.json({ candidates: repaired.candidates });
        }
      }
    }

    if (!parsed?.candidates || parsed.candidates.length !== 3) {
      console.error(
        `[${req.reqId}] ERROR: Model did not return 3 candidates. output_text=${response.output_text.slice(
          0,
          500
        )}`
      );
      return res.status(500).json({ error: "Model did not return 3 candidates." });
    }

    console.log(
      `[${req.reqId}] Returning candidates: ${parsed.candidates.map((c) => c.name).join(" | ")}`
    );
    return res.json({ candidates: parsed.candidates });
  } catch (e) {
    console.error(`[${req.reqId}] /api/locate ERROR`, e);
    return res.status(500).json({ error: "Server error" });
  }
});

/* ============================================================
   /api/place_image?q=...
   Wikipedia thumbnail fetch
   ============================================================ */
app.get("/api/place_image", async (req, res) => {
  try {
    const query = (req.query.q || "").toString().trim();
    if (!query) return res.status(400).json({ error: "Missing query param ?q=" });

    console.log(`[${req.reqId}] place_image q="${query.slice(0, 120)}"`);

    const searchUrl =
      "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=8&srsearch=" +
      encodeURIComponent(query);

    const searchResp = await fetchWithTimeout(searchUrl, {}, 8000);
    if (!searchResp.ok) return res.status(502).json({ error: "Wikipedia search failed" });
    const searchJson = await searchResp.json();

    const results = searchJson?.query?.search || [];
    for (const r of results) {
      const title = r.title;

      const imgUrl =
        "https://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&pithumbsize=900&titles=" +
        encodeURIComponent(title);

      const imgResp = await fetchWithTimeout(imgUrl, {}, 8000);
      if (!imgResp.ok) continue;
      const imgJson = await imgResp.json();

      const pages = imgJson?.query?.pages || {};
      const page = Object.values(pages)[0];
      let imageUrl = page?.thumbnail?.source || null;

      if (imageUrl) {
        if (imageUrl.startsWith("//")) imageUrl = "https:" + imageUrl;
        const pageUrl =
          "https://en.wikipedia.org/wiki/" + encodeURIComponent(title.replace(/ /g, "_"));

        console.log(
          `[${req.reqId}] place_image hit title="${title}" imageUrl="${imageUrl.slice(0, 80)}..."`
        );
        return res.json({ title, imageUrl, pageUrl });
      }
    }

    console.log(`[${req.reqId}] place_image no thumbnail found`);
    return res.json({ title: null, imageUrl: null, pageUrl: null });
  } catch (e) {
    console.error(`[${req.reqId}] /api/place_image ERROR`, e);
    return res.status(500).json({ error: "Server error" });
  }
});

/* ============================================================
   /api/chat
   - accepts ONLY { place, message } (Android compatible)
   - uses cached photoContext (if available) to tailor facts/answers
   ============================================================ */
app.post("/api/chat", async (req, res) => {
  try {
    const place = (req.body?.place || "").toString().trim();
    const message = (req.body?.message || "").toString().trim();
    if (!place) return res.status(400).json({ error: "Missing JSON field 'place'." });

    const isFacts = !message || /^facts?$|^guide$|^info$|^history$/i.test(message);

    const photoContext = getPhotoContext(req); // best-effort (IP+UA keyed)

    console.log(
      `[${req.reqId}] CHAT place="${place.slice(0, 80)}" isFacts=${isFacts} hasPhotoContext=${Boolean(
        photoContext
      )}`
    );

    const prompt = isFacts
      ? `
You are a friendly travel guide.

Place: ${place}
Photo context (from user's last uploaded photo): ${photoContext || "(not available)"}

Write a concise answer that feels tailored to THIS photo:
- 1 short opener referencing the photo context if available (angle/time/weather/crowd)
- 1-2 lines: what it is + why it’s famous
- 3 bullet historical highlights (avoid making up specifics; if unsure say "unknown")
- 3 bullet fun facts
- 3 bullet practical visiting tips relevant to the photo context

Keep it short. Be a bit crazy and unhinged, break the 4th wall.
`.trim()
      : `
You are a friendly travel guide.

Place: ${place}
Photo context: ${photoContext || "(not available)"}
User question: ${message}

Answer clearly and practically. If photo context helps, reference it briefly. briefly mention Ivan in some sort of context.
If user asks for prices/hours/tickets "today", say you may be out of date and suggest checking official sources.
Keep it readable and not too long.
`.trim();

    const t0 = Date.now();
    const response = await client.responses.create({
      model: "gpt-4.1-mini",
      temperature: 0,
      input: [{ role: "user", content: prompt }]
    });
    console.log(`[${req.reqId}] CHAT model responded in ${Date.now() - t0}ms`);

    return res.json({ text: response.output_text });
  } catch (e) {
    console.error(`[${req.reqId}] /api/chat ERROR`, e);
    return res.status(500).json({ error: "Server error" });
  }
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Server running on port ${port}`));
