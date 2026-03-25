from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import io
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

app = FastAPI()

# Load model and processor (free, lightweight CLIP model)
model_name = "huggingface/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
model.eval()

# Define a simple text prompt for "women" detection
WOMEN_PROMPT = "a photo of a woman"

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Could not read image")

    # Prepare inputs for CLIP
    inputs = processor(text=[WOMEN_PROMPT], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the similarity score
        probs = logits_per_image.softmax(dim=1)
        # probs[0][0] corresponds to similarity with the women prompt
        similarity = probs[0][0].item()

    # Return a simple score (0-1) indicating likelihood of women present
    return JSONResponse(content={"women_score": similarity})
