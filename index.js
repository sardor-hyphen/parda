const express = require('express');
const cors = require('cors');
const nsfwjs = require('nsfwjs');
const tf = require('@tensorflow/tfjs-node');
const axios = require('axios');

const app = express();
app.use(cors());
app.use(express.json());

let model;

// Load the model asynchronously when the server starts
// Using the official unpkg CDN to reliably load the default, lightweight quantized model
nsfwjs.load('https://unpkg.com/nsfwjs/example/nsfw_demo/public/quant_nsfw_mobilenet/').then((loadedModel) => {
    model = loadedModel;
    console.log('NSFWJS Model Loaded successfully.');
}).catch((err) => {
    console.error('Error loading NSFWJS model:', err);
});

// Health check endpoint
app.get('/', (req, res) => {
    res.json({ status: 'NSFW Filter API is running', modelLoaded: !!model });
});

// The main classification endpoint
app.post('/check', async (req, res) => {
    const { url } = req.body;
    if (!url) {
        return res.status(400).json({ error: 'Image URL is required' });
    }
    
    if (!model) {
        return res.status(503).json({ error: 'Model is still loading, please try again in a moment' });
    }

    try {
        // Fetch the image as a buffer
        const response = await axios.get(url, { responseType: 'arraybuffer' });
        const imageBuffer = Buffer.from(response.data, 'binary');

        // Decode the image into a TensorFlow 3D Tensor
        const imageTensor = tf.node.decodeImage(imageBuffer, 3);
        
        // Classify the image
        const predictions = await model.classify(imageTensor);
        
        // Clean up the tensor to prevent memory leaks
        imageTensor.dispose();

        // Return the array of predictions (e.g., Drawing, Hentai, Neutral, Porn, Sexy)
        res.json({ predictions });
    } catch (error) {
        console.error(`Error processing image URL ${url}:`, error.message);
        res.status(500).json({ error: 'Failed to process image', details: error.message });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`NSFW Filter Backend API is running on http://localhost:${PORT}`);
});
