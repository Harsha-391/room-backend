const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const { GoogleGenAI } = require("@google/genai");
require('dotenv').config();

const app = express();

// VERCEL TIP: Use MemoryStorage (Keep file in RAM, not on disk)
// Limit file size to 5MB and only allow images
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB limit
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed!'), false);
    }
  }
});

// 1. Setup Gemini
const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

app.get('/', (req, res) => {
  res.send('Room Visualizer Brain is Live on Vercel! 🧠');
});

app.post('/generate-room', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No image uploaded' });

    const materialName = req.body.material || "Marble";
    console.log(`Processing: ${materialName}`);

    // --- STEP A: GEMINI VISION ---
 // --- STEP A: GEMINI VISION (Enhanced) ---
    const base64Image = req.file.buffer.toString('base64');
    
    // Ask Gemini to create a perfect prompt for Stability AI
    const geminiResponse = await genAI.models.generateContent({
      model: 'gemini-1.5-flash',
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
    console.log(`🤖 Gemini Generated Prompt: ${optimizedPrompt}`);

    // --- STEP B: SEGMENTATION ---
    // Send the buffer directly (Faster!)
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
   // --- STEP C: STABILITY AI ---
    const formData = new FormData();
    formData.append('image', req.file.buffer, { filename: 'image.jpg' });
    formData.append('mask', maskBuffer, { filename: 'mask.png' });
    // Use the optimized prompt from Gemini
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

    res.set('Content-Type', 'image/png');
    res.send(stabilityResponse.data);

  } catch (error) {
    console.error("Error:", error.response ? error.response.data : error.message);
    res.status(500).json({ error: "Generation Failed" });
  }
});

// VERCEL TIP: Export the app, don't listen()
module.exports = app;