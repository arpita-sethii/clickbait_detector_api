from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os

app = FastAPI(title="Clickbait Detector API")

# Enable CORS for Chrome Extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
tokenizer = None
model = None
t5_tokenizer = None
t5_model = None
device = None

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global tokenizer, model, t5_tokenizer, t5_model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load DistilBERT
    print("Loading DistilBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "./clickbait_detector_model"
    )
    model.to(device)
    model.eval()
    print("✓ DistilBERT loaded")
    
    # Load T5
    print("Loading T5 model...")
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
    t5_model = T5ForConditionalGeneration.from_pretrained(
        "./t5_clickbait_rewriter_finetuned"
    )
    t5_model.to(device)
    t5_model.eval()
    print("✓ T5 loaded")

class HeadlineRequest(BaseModel):
    headline: str

@app.get("/")
def root():
    return {
        "name": "Clickbait Detector API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "/detect": "POST - Detect if headline is clickbait",
            "/rewrite": "POST - Rewrite clickbait to neutral",
            "/analyze": "POST - Detect + Rewrite in one call"
        }
    }

@app.post("/detect")
def detect_clickbait(request: HeadlineRequest):
    """Detect if headline is clickbait"""
    try:
        inputs = tokenizer(
            request.headline,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        
        return {
            "headline": request.headline,
            "is_clickbait": bool(prediction),
            "confidence": float(confidence),
            "label": "clickbait" if prediction else "neutral"
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/rewrite")
def rewrite_headline(request: HeadlineRequest):
    """Rewrite clickbait headline to neutral"""
    try:
        prompt = f"rewrite clickbait to neutral: {request.headline}"
        inputs = t5_tokenizer(
            prompt,
            return_tensors='pt',
            max_length=128,
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            outputs = t5_model.generate(
                inputs['input_ids'],
                max_length=64,
                num_beams=5,
                early_stopping=True
            )
        
        neutral = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "original": request.headline,
            "rewritten": neutral
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/analyze")
def analyze_headline(request: HeadlineRequest):
    """Detect and rewrite in one call"""
    try:
        detection = detect_clickbait(request)
        
        if detection.get("is_clickbait"):
            rewrite = rewrite_headline(request)
            return {
                **detection,
                "rewritten": rewrite.get("rewritten")
            }
        else:
            return {
                **detection,
                "rewritten": request.headline,
                "message": "Headline is already neutral"
            }
    except Exception as e:
        return {"error": str(e)}
