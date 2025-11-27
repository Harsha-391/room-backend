const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const { GoogleGenAI } = require("@google/genai");
const mongoose = require('mongoose'); // <--- 1. Import Mongoose
require('dotenv').config();

const app = express();

// Limit file size to 5MB and only allow images
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 }, 
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'), false);
    }
  }
});

app.use(cors());
app.use(express.json());

// --- 2. DATABASE CONNECTION ---
// Only connect if the URI is present
if (process.env.MONGODB_URI) {
  mongoose.connect(process.env.MONGODB_URI)
    .then(() => console.log("✅ Connected to MongoDB"))
    .catch(err => console.error("❌ MongoDB Connection Error:", err));
} else {
  console.warn("⚠️ MONGODB_URI is missing in .env file");
}

// --- 3. DEFINE THE DATA STRUCTURE (SCHEMA) ---
const RoomSchema = new mongoose.Schema({
  originalPrompt: String,
  optimizedPrompt: String,
  material: String,
  createdAt: { type: Date, default: Date.now },
  imageBase64: String, // We will store the final image string here
});

// Create the model
const Room = mongoose.model('Room', RoomSchema);

// Setup Gemini
const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

app.get('/', (req, res) => {
  res.send('Room Visualizer Brain is Live on Vercel! 🧠');
});

// --- 4. HISTORY ENDPOINT ---
// Call this from your app to see past designs!
app.get('/history', async (req, res) => {
  try {
    // Get the last 20 generated rooms, newest first
    const history = await Room.find().sort({ createdAt: -1 }).limit(20);
    res.json(history);
  } catch (error) {
    console.error("History Error:", error);
    res.status(500).json({ error: "Could not fetch history" });
  }
});

app.post('/generate-room', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No image uploaded' });

    const materialName = req.body.material || "Marble";
    console.log(`Processing: ${materialName}`);

    // --- STEP A: GEMINI VISION (Better Prompting) ---
    const base64Image = req.file.buffer.toString('base64');
    
    const geminiResponse = await genAI.models.generateContent({
      model: 'gemini-2.0-flash-vision',
      contents: [
        { inlineData: { mimeType: req.file.mimetype, data: base64Image } },
        { text: `The user wants to replace the flooring in this room with: "${materialName}". 
          Analyze the room's lighting (shadows, temperature), perspective, and style.
          Write a concise, high-quality text prompt for an inpainting model (like Stable Diffusion) to generate this new floor photorealistically. 
          Mention the lighting interaction on the floor. 
          Output ONLY the prompt text.` }
      ]
    });
    
    const optimizedPrompt = geminiResponse.response.text().trim();
    console.log(`🤖 Prompt: ${optimizedPrompt}`);

    // --- STEP B: SEGMENTATION ---
    const maskResponse = await axios.post(
      "https://api-inference.huggingface.co/models/nvidia/segformer-b0-finetuned-ade-512-512",
      req.file.buffer, 
      {
        headers: { 
            Authorization: `Bearer ${process.env.HUGGINGFACE_TOKEN}`,
            "Content-Type": "application/octet-stream"
        },
        responseType: 'arraybuffer' 
      }
    );
    const maskBuffer = Buffer.from(maskResponse.data);

    // --- STEP C: STABILITY AI ---
    const formData = new FormData();
    formData.append('image', req.file.buffer, { filename: 'image.jpg' });
    formData.append('mask', maskBuffer, { filename: 'mask.png' });
    formData.append('prompt', `${optimizedPrompt} . high quality, 8k, photorealistic, interior design`);
    formData.append('output_format', 'png');

    const stabilityResponse = await axios.post(
        'https://api.stability.ai/v2beta/stable-image/edit/inpaint',
        formData,
        {
            headers: {
                ...formData.getHeaders(),
                Authorization: `Bearer ${process.env.STABILITY_API_KEY}`,
                Accept: 'image/*'
            },
            responseType: 'arraybuffer'
        }
    );

    const finalImageBase64 = Buffer.from(stabilityResponse.data).toString('base64');
    const finalDataURI = `data:image/png;base64,${finalImageBase64}`;

    // --- 5. SAVE TO DATABASE ---
    try {
        if (process.env.MONGODB_URI) {
            const newRoom = await Room.create({
                originalPrompt: materialName,
                optimizedPrompt: optimizedPrompt,
                material: materialName,
                imageBase64: finalDataURI
            });
            console.log(`💾 Saved to DB: ${newRoom._id}`);
        }
    } catch (dbError) {
        console.error("⚠️ Database Save Failed (Image returned anyway):", dbError);
    }

    // Return success even if DB fails (User gets their image)
    res.status(200).json({
        success: true,
        data: finalDataURI,
        promptUsed: optimizedPrompt,
        message: "Room generated successfully!"
    });

  } catch (error) {
    console.error("Detailed Error:", error);
    const errorMessage = error.response?.data?.error || error.message || "Unknown Server Error";
    res.status(500).json({ 
      error: "Generation Failed", 
      details: errorMessage 
    });
  }
});

module.exports = app;