---
title: Clickbait Detector API
emoji: 🎯
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
---

# Clickbait Detector API

AI-powered API to detect and neutralize clickbait headlines using DistilBERT and T5.

## Endpoints

- `GET /` - API info
- `POST /detect` - Detect if headline is clickbait
- `POST /rewrite` - Rewrite clickbait to neutral
- `POST /analyze` - Detect + rewrite in one call

## Usage
```python
import requests

response = requests.post(
    "https://YOUR-SPACE.hf.space/detect",
    json={"headline": "You Won't Believe What Happened!"}
)
print(response.json())
```

Powered by DistilBERT and T5.
