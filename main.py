from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import uvicorn
import shutil
import os
import re

from app.core.config import settings
from app.services import embedding_service, llm_service, vector_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Portfolio RAG Backend (OpenRouter + Pinecone)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    model_provider: str = "openrouter"
    history: List[ChatMessage] = []

# Project keyword mapping for SMART media retrieval
PROJECT_KEYWORDS = {
    "3d-mri-sr": ["mri", "3d mri", "super resolution", "medical imaging", "brain scan", "mri video", "mri project"],
    "verdex-mobile-app": ["verdex", "crop disease", "farmer", "mobile app", "agriculture"],
    "diabetic-retinopathy": ["diabetic", "retinopathy", "fundus", "eye disease"],
    "melanoma-classification": ["melanoma", "skin cancer", "dermoscopic", "siim-isic"],
    "sinhala-rag": ["sinhala fact", "fact-checking", "misinformation", "sinhala rag"],
    "sinhala-idioms": ["idioms", "sinhala idioms", "translation"],
}

# STRICT keywords for showing profile image
PROFILE_KEYWORDS = ['who are you', 'about yourself', 'your photo', 'your picture', 'your image', 'show yourself']

def get_matched_project_ids(query: str) -> set:
    """Find which projects are mentioned in the query."""
    query_lower = query.lower()
    matched = set()
    for project_id, keywords in PROJECT_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                matched.add(project_id)
                break
    return matched

def should_show_profile_image(query: str) -> bool:
    """STRICT check - only if user explicitly asks about the person/photo."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in PROFILE_KEYWORDS)

def should_show_media(query: str) -> bool:
    """Check if user wants to see visuals/media."""
    media_words = ['show', 'image', 'picture', 'photo', 'video', 'demo', 'visual', 'see', 'watch']
    query_lower = query.lower()
    return any(w in query_lower for w in media_words)

def clean_suggestion(suggestion: str) -> str:
    s = suggestion.replace('**', '').replace('*', '').replace('"', '')
    s = re.sub(r'^\d+\.\s*', '', s)
    s = re.sub(r'^-\s*', '', s)
    return s.strip()

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Backend is running with OpenRouter + Pinecone"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        query = request.message
        logger.info(f"Received query: {query}")

        # Determine intent
        matched_projects = get_matched_project_ids(query)
        show_profile = should_show_profile_image(query)
        wants_media = should_show_media(query)
        
        logger.info(f"Intent Analysis: matched_projects={matched_projects}, show_profile={show_profile}, wants_media={wants_media}")

        # Embed Query
        try:
            embedding = embedding_service.get_embedding(query)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding Error: {str(e)}")
        
        # Retrieve Context
        search_results = vector_db.query_vectors(embedding, top_k=100)
        
        context_text = ""
        media_by_project = {}
        profile_image = None

        for match in search_results.get('matches', []):
            score = match['score']
            metadata = match.get('metadata', {})
            text = metadata.get('text', '')
            match_type = metadata.get('type', '')
            project_id = metadata.get('project_id', '')
            
            if score > 0.25:
                context_text += text + "\n---\n"
                
                # Collect profile image (but only USE it if show_profile is True)
                if match_type == 'profile' and 'image_url' in metadata:
                    profile_image = metadata['image_url']
                
                # Collect project media
                if match_type == 'project' and project_id:
                    if project_id not in media_by_project:
                        media_by_project[project_id] = {
                            'images': [],
                            'videos': [],
                            'title': metadata.get('title', 'Project')
                        }
                    
                    if 'image_url' in metadata and metadata['image_url']:
                        media_by_project[project_id]['images'].append(metadata['image_url'])
                    if 'video_url' in metadata and metadata['video_url']:
                        media_by_project[project_id]['videos'].append(metadata['video_url'])

        logger.info(f"Collected media for projects: {list(media_by_project.keys())}")

        # Generate Answer
        try:
            full_response = llm_service.generate_response(query, context_text, "")
            
            response_text = full_response
            suggestions = []
            
            if "<<SUGGESTIONS>>" in full_response:
                parts = full_response.split("<<SUGGESTIONS>>")
                response_text = parts[0].strip()
                suggestions_raw = parts[1].strip().split("\n")
                suggestions = [clean_suggestion(s) for s in suggestions_raw if s.strip()]
            
            # SMART MEDIA ATTACHMENT
            final_images = []
            final_videos = []
            
            # Profile image: ONLY if explicitly asked
            if show_profile and profile_image:
                logger.info("Adding profile image (explicitly requested)")
                final_images.append(profile_image)
            
            # Project media:
            # 1. If user explicitly wants media, show all relevant images/videos
            # 2. If project is STRONGLY matched in context (high score), show video even without explicit request
            if wants_media:
                for project_id in matched_projects:
                    if project_id in media_by_project:
                        media = media_by_project[project_id]
                        logger.info(f"Adding media for project: {project_id}")
                        final_images.extend(media['images'])
                        final_videos.extend(media['videos'])
            else:
                # Fallback: Check if any project was strongly matched in retrieval (automatic video suggestion)
                for project_id, media in media_by_project.items():
                    if media['videos']:
                        logger.info(f"Auto-suggesting video for relevant project: {project_id}")
                        final_videos.extend(media['videos'])

            
            # Deduplicate
            final_images = list(dict.fromkeys(final_images))
            final_videos = list(dict.fromkeys(final_videos))
            
            # Append to response text
            if final_images or final_videos:
                response_text += "\n\n**Visuals:**\n"
                
                for img in final_images:
                    if img not in response_text:
                        response_text += f"![Visual]({img})\n"
                
                for vid in final_videos:
                    if vid not in response_text:
                        response_text += f"\n🎥 **Watch Video:** [Project Demo]({vid})\n"

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

        return {
            "response": response_text,
            "provider": "openrouter",
            "context_used": len(context_text) > 0,
            "images": final_images,
            "videos": final_videos,
            "suggestions": suggestions
        }

    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
