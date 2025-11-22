from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv

## break vercel deployment
# import clip
# import torch
# import numpy as np
# import faiss
# from pathlib import Path
# from PIL import Image
# import prophet

load_dotenv()

app = FastAPI()

# CORS so the frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str
    mood: str = ""  # User's selected mood from turkey selection

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/api/chat")
def chat(request: ChatRequest):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    try:
        user_message = request.message
        mood = request.mood or ""
        
        # Debug logging
        print(f"[DEBUG] Received request - Mood: '{mood}' (length: {len(mood)})")
        print(f"[DEBUG] Full request body: message length={len(user_message)}, mood='{mood}'")
        
        # Build system prompt with mood information
        system_prompt = "You are the HotMessCoach - a supportive, funny, and sarcastic Thanksgiving survival coach. "
        
        if mood and mood.strip():
            # Format mood name nicely (capitalize first letter of each word)
            mood_formatted = mood.strip().replace("_", " ").title()
            system_prompt += f"The user is currently feeling {mood_formatted} - keep this in mind when responding with empathy and understanding. "
            print(f"[DEBUG] Mood included in system prompt: {mood_formatted}")
        else:
            print("[DEBUG] WARNING: No mood received or mood is empty!")
        
        system_prompt += "Be funny, sarcastic, but genuinely helpful. Use turkey and food emojis. Keep responses concise and entertaining!"
        
        print(f"[DEBUG] System prompt: {system_prompt[:200]}...")
        
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")
