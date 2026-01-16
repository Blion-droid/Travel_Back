import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI from "openai";

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

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
   /api/locate
   - Generates photoContext + candidates
   - Caches photoContext server-side
   - Returns ONLY candidates (so Android doesn't need changes)
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

    const locationBlock = hasLoc
      ? `Location is available and HIGH PRIORITY.
Coordinates: lat=${lat}, lon=${lon}${Number.isFinite(acc) ? ` (accuracy ≈ ${acc}m)` : ""}.
Strongly prefer candidates plausible near these coordinates, unless the photo clearly contradicts it (then mention conflict and lower confidence).`
      : `Location is NOT available. Rely on visual cues + user text.`;

    // Model returns photoContext + candidates, but we only return candidates to the client.
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
Describe what is visible in THIS photo (viewpoint, lighting/time, weather if visible, materials, architecture, signs, crowd level, surroundings).
Do NOT name/guess the place in this section.

Task B — IDENTIFY PLACE:
Return EXACTLY 3 candidates ranked best -> worst.
For each candidate:
- name: short friendly place name WITH city/country if possible
- why: 1–2 sentences referencing visual cues AND (if provided) how coordinates affected the choice
- confidence: 0..1
- searchQuery: likely query to find a representative image

${locationBlock}

User request text:
${userText ? userText : "(none)"}

Rules:
- Don't invent certainty.
- If photo contradicts coords, mention it in "why" and lower confidence.
`.trim();

    const response = await client.responses.create({
      model: "gpt-4.1-mini",
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

    const parsed = JSON.parse(response.output_text);

    // Cache photoContext for later /api/chat calls (no Android changes required)
    if (parsed?.photoContext) setPhotoContext(req, parsed.photoContext);

    // Return only what your existing Android app expects:
    // { candidates: [...] }
    if (!parsed?.candidates || parsed.candidates.length !== 3) {
      return res.status(500).json({ error: "Model did not return 3 candidates." });
    }

    return res.json({ candidates: parsed.candidates });
  } catch (e) {
    console.error(e);
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

    const searchUrl =
      "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=8&srsearch=" +
      encodeURIComponent(query);

    const searchResp = await fetch(searchUrl);
    if (!searchResp.ok) return res.status(502).json({ error: "Wikipedia search failed" });
    const searchJson = await searchResp.json();

    const results = searchJson?.query?.search || [];
    for (const r of results) {
      const title = r.title;

      const imgUrl =
        "https://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&pithumbsize=900&titles=" +
        encodeURIComponent(title);

      const imgResp = await fetch(imgUrl);
      if (!imgResp.ok) continue;
      const imgJson = await imgResp.json();

      const pages = imgJson?.query?.pages || {};
      const page = Object.values(pages)[0];
      let imageUrl = page?.thumbnail?.source || null;

      if (imageUrl) {
        if (imageUrl.startsWith("//")) imageUrl = "https:" + imageUrl;
        const pageUrl =
          "https://en.wikipedia.org/wiki/" + encodeURIComponent(title.replace(/ /g, "_"));
        return res.json({ title, imageUrl, pageUrl });
      }
    }

    return res.json({ title: null, imageUrl: null, pageUrl: null });
  } catch (e) {
    console.error(e);
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

    const prompt = isFacts
      ? `
You are a friendly travel guide.

Place: ${place}
Photo context (from user's last uploaded photo): ${photoContext || "(not available)"}

Write a concise answer that feels tailored to THIS photo:
- 1 short opener referencing the photo context if available (angle/time/weather/crowd)
- 2–3 lines: what it is + why it’s famous
- 3 bullet historical highlights (avoid making up specifics; if unsure say "unknown")
- 3 bullet fun facts
- 3 bullet practical visiting tips relevant to the photo context

Keep it readable, well formated and short-medium length. Be a bit crazy and unhinged, break the 4th wall.
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

    const response = await client.responses.create({
      model: "gpt-4.1-mini",
      input: [{ role: "user", content: prompt }]
    });

    return res.json({ text: response.output_text });
  } catch (e) {
    console.error(e);
    return res.status(500).json({ error: "Server error" });
  }
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Server running on port ${port}`));
