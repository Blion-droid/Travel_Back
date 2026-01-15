import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI from "openai";

const app = express();
app.use(cors());

// JSON body for /api/facts
app.use(express.json({ limit: "1mb" }));

// Multer in-memory (good for Render)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
});

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.get("/health", (_, res) => res.json({ ok: true }));

/**
 * POST /api/locate
 * multipart/form-data:
 *   - image: file
 *   - message: string (optional)
 *   - lat: string/number (optional)
 *   - lon: string/number (optional)
 *   - accuracyMeters: string/number (optional)
 *
 * returns JSON:
 *   { candidates: [ { name, why, confidence, searchQuery }, ... x3 ] }
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
        `Device location hint (use as a hint only): lat=${lat}, lon=${lon}` +
          (Number.isFinite(acc) ? ` (accuracy ≈ ${acc}m)` : "")
      );
    }
    if (userText) hints.push(`User request: ${userText}`);

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
Goal: Guess what location/place is in the photo.

Return EXACTLY 3 candidates ranked best -> worst.
For each candidate:
- name: short friendly name (e.g. "Colosseum, Rome, Italy")
- why: 1–2 sentences describing visual cues (and if uncertain, say so)
- confidence: number 0..1
- searchQuery: a query string that would find a representative photo (use the name)

Rules:
- Do NOT invent certainty. If unsure, say so in "why".
- If location hint is present, treat it as a hint only (GPS may be wrong).
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

    // When using json_schema, output_text is a JSON string
    const parsed = JSON.parse(response.output_text);

    // Basic sanity
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
 * Uses Wikipedia API to find a page and return a thumbnail image URL.
 * This is a safe alternative to scraping Google Images.
 */
app.get("/api/place_image", async (req, res) => {
  try {
    const query = (req.query.q || "").toString().trim();
    if (!query) return res.status(400).json({ error: "Missing query param ?q=" });

    // 1) Search Wikipedia
    const searchUrl =
      "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srsearch=" +
      encodeURIComponent(query);

    const searchResp = await fetch(searchUrl);
    if (!searchResp.ok) return res.status(502).json({ error: "Wikipedia search failed" });
    const searchJson = await searchResp.json();

    const first = searchJson?.query?.search?.[0];
    if (!first?.title) return res.json({ title: null, imageUrl: null, pageUrl: null });

    const title = first.title;

    // 2) Get page image thumbnail
    const imgUrl =
      "https://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&pithumbsize=700&titles=" +
      encodeURIComponent(title);

    const imgResp = await fetch(imgUrl);
    if (!imgResp.ok) return res.status(502).json({ error: "Wikipedia image lookup failed" });
    const imgJson = await imgResp.json();

    const pages = imgJson?.query?.pages || {};
    const page = Object.values(pages)[0];
    const imageUrl = page?.thumbnail?.source || null;

    const pageUrl = "https://en.wikipedia.org/wiki/" + encodeURIComponent(title.replace(/ /g, "_"));
    res.json({ title, imageUrl, pageUrl });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Server error" });
  }
});

/**
 * POST /api/facts
 * JSON: { place: string, message?: string }
 * returns: { text: string }
 */
app.post("/api/facts", async (req, res) => {
  try {
    const place = (req.body?.place || "").toString().trim();
    const userText = (req.body?.message || "").toString().trim();

    if (!place) return res.status(400).json({ error: "Missing JSON field 'place'." });

    const prompt = `
You are a friendly travel guide.

Place: ${place}
User request: ${userText || "(none)"}

Write a concise chat-style answer:
- 2–3 lines: what it is + why it’s famous
- 3 bullet historical highlights (avoid making up specifics; if unsure say "unknown")
- 3 bullet fun facts
- 3 bullet practical visiting tips (time of day, tickets, etiquette, safety)
Keep it readable and not too long.
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
