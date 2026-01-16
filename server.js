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

app.get("/health", (_, res) => res.json({ ok: true }));

/**
 * POST /api/locate
 * multipart/form-data:
 *   image: file  (required, field name MUST be "image")
 *   message: string (optional)
 *   lat, lon, accuracyMeters: optional strings/numbers
 *
 * returns:
 * { candidates: [ { name, why, confidence, searchQuery }, ... x3 ] }
 */
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

    const hints = [];
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
      hints.push(
        `Device location hint (hint only, may be wrong): lat=${lat}, lon=${lon}` +
          (Number.isFinite(acc) ? ` (accuracy ≈ ${acc}m)` : "")
      );
    }
    if (userText) hints.push(`User request: ${userText}`);

    // Strict JSON schema for stable parsing
    const schema = {
      type: "object",
      properties: {
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
      required: ["candidates"],
      additionalProperties: false
    };

    const prompt = `
You are a travel expert.
Goal: Identify the location/place in the photo.

Return EXACTLY 3 candidates ranked best -> worst.
For each candidate:
- name: short friendly name (e.g. "Colosseum, Rome, Italy")
- why: 1–2 sentences describing visual cues; if unsure, say so
- confidence: number 0..1
- searchQuery: query string to find a representative photo (usually same as name)

Rules:
- Do NOT invent certainty.
- If a device location hint is present, treat it as a hint only.
${hints.length ? "\nHints:\n- " + hints.join("\n- ") : ""}
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
          name: "location_candidates",
          strict: true,
          schema
        }
      }
    });

    const parsed = JSON.parse(response.output_text);
    if (!parsed?.candidates || parsed.candidates.length !== 3) {
      return res.status(500).json({ error: "Model did not return 3 candidates." });
    }

    res.json(parsed);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Server error" });
  }
});

/**
 * GET /api/place_image?q=...
 * Uses Wikipedia API to find a page with a thumbnail.
 * (More reliable than scraping Google; safe for prototypes.)
 *
 * returns:
 * { title, imageUrl, pageUrl }
 */
app.get("/api/place_image", async (req, res) => {
  try {
    const query = (req.query.q || "").toString().trim();
    if (!query) return res.status(400).json({ error: "Missing query param ?q=" });

    const searchUrl =
      "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=6&srsearch=" +
      encodeURIComponent(query);

    const searchResp = await fetch(searchUrl);
    if (!searchResp.ok) return res.status(502).json({ error: "Wikipedia search failed" });
    const searchJson = await searchResp.json();

    const results = searchJson?.query?.search || [];

    for (const r of results) {
      const title = r.title;

      const imgUrl =
        "https://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&pithumbsize=800&titles=" +
        encodeURIComponent(title);

      const imgResp = await fetch(imgUrl);
      if (!imgResp.ok) continue;
      const imgJson = await imgResp.json();

      const pages = imgJson?.query?.pages || {};
      const page = Object.values(pages)[0];
      let imageUrl = page?.thumbnail?.source || null;

      if (imageUrl) {
        // normalize protocol-relative urls
        if (imageUrl.startsWith("//")) imageUrl = "https:" + imageUrl;
        const pageUrl =
          "https://en.wikipedia.org/wiki/" + encodeURIComponent(title.replace(/ /g, "_"));
        return res.json({ title, imageUrl, pageUrl });
      }
    }

    return res.json({ title: null, imageUrl: null, pageUrl: null });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Server error" });
  }
});

/**
 * POST /api/chat
 * JSON: { place: string, message: string }
 *
 * Used for:
 * - facts (when message is empty or "facts")
 * - follow-up questions
 *
 * returns: { text: string }
 */
app.post("/api/chat", async (req, res) => {
  try {
    const place = (req.body?.place || "").toString().trim();
    const message = (req.body?.message || "").toString().trim();

    if (!place) return res.status(400).json({ error: "Missing JSON field 'place'." });

    const mode =
      !message || /^facts?$|^guide$|^info$|^history$/i.test(message) ? "FACTS" : "QA";

    const prompt =
      mode === "FACTS"
        ? `
You are a friendly travel guide.
Place: ${place}

Write a concise chat-style answer:
- 2–3 lines: what it is + why it’s famous
- 3 bullet historical highlights (don’t invent specifics; if unsure say "unknown")
- 3 bullet fun facts
- 3 bullet practical visiting tips
Keep it readable and not too long, also say something personal to the user to break the 4th wall, be unhinged, sometimes act crazy.
`.trim()
        : `
You are a friendly travel guide.
Place: ${place}

User question: ${message}

Answer clearly and practically, add some rare rude comments, its a sort of game. be unhinged, sometimes act crazy - never admit it.
If the user asks for prices/hours/tickets "today", say you may be out of date and suggest checking the official site.
`.trim();

    const response = await client.responses.create({
      model: "gpt-4.1-mini",
      input: [{ role: "user", content: prompt }]
    });

    res.json({ text: response.output_text });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Server error" });
  }
});

const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Server running on port ${port}`));
