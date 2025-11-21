const express = require('express');
const cors = require('cors');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const { GoogleGenAI } = require("@google/genai");
require('dotenv').config();

const app = express();
const upload = multer({ dest: 'uploads/' });

app.use(cors());
app.use(express.json());

// 1. Setup Gemini (The Vision AI)
const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

app.get('/', (req, res) => {
  res.send('Room Visualizer Backend is Live! 🚀');
});

// THE MAIN ENDPOINT
app.post('/generate-room', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No image uploaded' });
    
    const filePath = req.file.path;
    const materialName = req.body.material || "Marble";

    console.log(`1. Processing image for material: ${materialName}...`);

    // --- STEP A: ASK GEMINI TO DESCRIBE THE ROOM ---
    // We use Gemini 1.5 Flash (cheapest & fastest)
    const imageBuffer = fs.readFileSync(filePath);
    const base64Image = imageBuffer.toString('base64');
    
    const geminiResponse = await genAI.models.generateContent({
      model: 'gemini-1.5-flash',
      contents: [
        {
          inlineData: {
            mimeType: req.file.mimetype,
            data: base64Image
          }
        },
        { text: "Describe this room briefly. Mention the lighting, the style, and where the floor is." }
      ]
    });
    
    const description = geminiResponse.response.text(); 
    console.log("Gemini says:", description);

    // --- STEP B: GET SEGMENTATION MASK (Hugging Face) ---
    console.log("2. Generating Floor Mask...");
    const maskResponse = await axios.post(
      "https://api-inference.huggingface.co/models/nvidia/segformer-b0-finetuned-ade-512-512",
      imageBuffer,
      {
        headers: { 
            Authorization: `Bearer ${process.env.HUGGINGFACE_TOKEN}`,
            "Content-Type": "application/octet-stream"
        },
        responseType: 'arraybuffer' 
      }
    );
    const maskBuffer = Buffer.from(maskResponse.data);

    // --- STEP C: GENERATE FINAL IMAGE (Stability AI) ---
    console.log("3. Inpainting with Stability AI...");
    const formData = new FormData();
    formData.append('image', fs.createReadStream(filePath));
    formData.append('mask', maskBuffer, { filename: 'mask.png' });
    formData.append('prompt', `Replace the floor with ${materialName}. ${description}. Photorealistic, high gloss, 8k resolution.`);
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

    // Cleanup and Send
    fs.unlinkSync(filePath); // Delete temp file
    console.log("Success! Sending image back.");
    res.set('Content-Type', 'image/png');
    res.send(stabilityResponse.data);

  } catch (error) {
    console.error("Error:", error.response ? error.response.data : error.message);
    res.status(500).json({ error: "Generation Failed" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));