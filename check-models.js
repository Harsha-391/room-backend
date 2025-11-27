const { GoogleGenAI } = require("@google/genai");
require('dotenv').config();

const genAI = new GoogleGenAI({ apiKey: process.env.AIzaSyCWDmuTEaJUtlyXLeZxI7fb1sivWz5WKIU });

async function listModels() {
  try {
    console.log("Fetching available models...");
    const response = await genAI.models.list();
    
    console.log("\n✅ AVAILABLE MODELS:");
    // Filter for models that support 'generateContent'
    response.models
      .filter(m => m.supportedGenerationMethods.includes('generateContent'))
      .forEach(m => console.log(`- ${m.name.replace('models/', '')}`)); // Clean name for you
      
  } catch (error) {
    console.error("❌ Error listing models:", error.message);
  }
}

listModels();