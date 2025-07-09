# gemini_canvas_app.py
# FastAPI app for text processing using Gemini, DeepSeek, and OpenAI (via OpenRouter)

import os
import logging
import dotenv
import google.generativeai as genai
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from datetime import datetime, timedelta
from fastapi.middleware.cors import CORSMiddleware

# Remove xai_sdk imports and Grok

dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Remove XAI_API_KEY and GROK_API_KEY
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set")
    raise ValueError("Please set GEMINI_API_KEY environment variable")
if not OPENROUTER_API_KEY:
    logger.error("OPENROUTER_API_KEY environment variable not set")
    raise ValueError("Please set OPENROUTER_API_KEY environment variable")

# Remove XAI_API_KEY check

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI(title="Gemini, DeepSeek, OpenAI Text Processor", version="1.0.0")

# Configure CORS more securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",         # Added for Vite/React dev server
        "http://127.0.0.1:5173",          # Added for Vite/React dev server
        "https://i20frontend.vercel.app"  # Added for deployed frontend
    ],  # Specify allowed origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str
    action: str  # 'rewrite', 'summarize', 'expand', 'improve', 'simplify'
    style: Optional[str] = None  # 'formal', 'casual', 'professional', 'creative'
    model: str  # 'gemini', 'deepseek', or 'openai'

def get_gemini_response(prompt: str):
    response = gemini_model.generate_content(prompt)
    class GeminiResponse:
        text = response.text
    return GeminiResponse()

def get_deepseek_response(prompt: str):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    try:
        resp = httpx.post(url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        text = result["choices"][0]["message"]["content"]
        class DeepSeekResponse:
            pass
        deepseek_response = DeepSeekResponse()
        deepseek_response.text = text
        return deepseek_response
    except httpx.HTTPStatusError as e:
        logger.error(f"DeepSeek API error: {e.response.text}")
        raise HTTPException(status_code=500, detail=f"DeepSeek API error: {e.response.text}")
    except Exception as e:
        logger.error(f"DeepSeek API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"DeepSeek API error: {str(e)}")

def get_openai_response(prompt: str):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    try:
        resp = httpx.post(url, headers=headers, json=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        text = result["choices"][0]["message"]["content"]
        class OpenAIResponse:
            pass
        openai_response = OpenAIResponse()
        openai_response.text = text
        return openai_response
    except httpx.HTTPStatusError as e:
        logger.error(f"OpenAI API error: {e.response.text}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e.response.text}")
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

@app.post("/process")
async def process_text(req: TextRequest):
    start_time_utc = datetime.utcnow()
    IST_OFFSET = timedelta(hours=5, minutes=30)
    start_time_ist = start_time_utc + IST_OFFSET
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    style_modifier = f" Make the tone {req.style}." if req.style else ""
    prompt_map = {
        "rewrite": f"Rewrite the following text in a clear, concise manner{style_modifier}:\n\n{req.text}",
        "summarize": f"Summarize the following text concisely{style_modifier}:\n\n{req.text}",
        "expand": f"Expand the following into more detailed content{style_modifier}:\n\n{req.text}",
        "improve": f"Improve the following text for clarity and engagement{style_modifier}:\n\n{req.text}",
        "simplify": f"Simplify the following text to make it easier to understand{style_modifier}:\n\n{req.text}",
    }
    prompt = prompt_map.get(req.action)
    if not prompt:
        raise HTTPException(status_code=400, detail=f"Unsupported action: {req.action}")

    try:
        match req.model.lower():
            case "gemini":
                response = get_gemini_response(prompt)
            case "deepseek":
                response = get_deepseek_response(prompt)
            case "openai":
                response = get_openai_response(prompt)
            case _:
                raise HTTPException(status_code=400, detail=f"Unsupported model: {req.model}")
        if not response.text:
            raise HTTPException(status_code=500, detail="No response generated")
        end_time_utc = datetime.utcnow()
        end_time_ist = end_time_utc + IST_OFFSET
        elapsed = (end_time_utc - start_time_utc).total_seconds()
        time_format = "%H:%M:%S, %d-%m-%Y"
        return {
            "result": response.text,
            "action": req.action,
            "style": req.style,
            "model": req.model,
            "start_time": start_time_ist.strftime(time_format),
            "end_time": end_time_ist.strftime(time_format),
            "elapsed_seconds": elapsed
        }
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app4:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )