import express from "express";
import multer from "multer";
import OpenAI from "openai";

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

app.get("/health", (req, res) => res.json({ ok: true }));

app.post("/api/guide", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "Missing form field 'image'." });

    const mime = req.file.mimetype || "image/jpeg";
    const base64 = req.file.buffer.toString("base64");
    const dataUrl = `data:${mime};base64,${base64}`;

    const prompt = "Identify this place and give history + fun facts + visiting tips.";

    const response = await client.responses.create({
      model: "gpt-4.1-mini",
      input: [{
        role: "user",
        content: [
          { type: "input_text", text: prompt },
          { type: "input_image", image_url: dataUrl }
        ]
      }]
    });

    res.json({ text: response.output_text });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: "Server error" });
  }
});

app.listen(process.env.PORT || 3000);
