"""
Portfolio Data Ingestion Script (OpenCLIP + Pinecone)
Loads portfolio data from JSON and creates embeddings for RAG retrieval.
Uses OpenCLIP for multimodal embeddings (text + images in same vector space).
"""
import time
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from app.services.clip_service import get_text_embedding, get_image_embedding
from app.services.vector_db import upsert_vectors, delete_all_vectors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "portfolio.json"

def load_portfolio_data():
    """Load and parse portfolio JSON data."""
    with open(DATA_PATH, 'r') as f:
        return json.load(f)

def create_text_chunks(data):
    """
    Convert portfolio data into text chunks for embedding.
    Each chunk has an id, text content, and metadata.
    """
    chunks = []
    
    # Profile chunk
    profile = data.get('profile', {})
    profile_text = f"""
    Name: {profile.get('name', 'Unknown')}
    Title: {profile.get('title', '')}
    Bio: {profile.get('bio', '')}
    Location: {profile.get('location', '')}
    """
    chunks.append({
        "id": "profile",
        "text": profile_text.strip(),
        "type": "profile",
        "metadata": {
            "section": "about",
            "image_url": profile.get('avatar_image')
        }
    })
    
    # Skills chunk
    skills = data.get('skills', {})
    skills_text = f"""
    Technical Skills:
    - Programming Languages: {', '.join(skills.get('languages', []))}
    - AI/ML: {', '.join(skills.get('ai_ml', []))}
    - Frameworks & Libraries: {', '.join(skills.get('frameworks_libraries', []))}
    - Development Platforms: {', '.join(skills.get('development_platforms', []))}
    - Cloud: {', '.join(skills.get('cloud', []))}
    """
    chunks.append({
        "id": "skills",
        "text": skills_text.strip(),
        "type": "skills",
        "metadata": {"section": "skills"}
    })
    
    # Project chunks 
    for project in data.get('projects', []):
        project_text = f"""
        Project: {project.get('title', '')}
        Description: {project.get('description', '')}
        Details: {project.get('long_description', '')}
        Technologies: {', '.join(project.get('tech_stack', []))}
        Features: {', '.join(project.get('features', []))}
        Status: {project.get('status', '')}
        Category: {project.get('category', '')}
        """
        
        if 'documents' in project:
            docs_str = "\n".join([f"- {d['name']} ({d['type']}): {d['url']}" for d in project['documents']])
            project_text += f"\nDocuments:\n{docs_str}\n"
        
        chunks.append({
            "id": f"project-{project.get('id', 'unknown')}",
            "text": project_text.strip(),
            "type": "project",
            "metadata": {
                "section": "projects",
                "project_id": project.get('id'),
                "title": project.get('title'),
                "demo_url": project.get('demo_url'),
                "github_url": project.get('github_url'),
                "image_url": project.get('image'),
                "video_url": project.get('video')
            }
        })
    
    # Experience chunks
    for exp in data.get('experience', []):
        exp_text = f"""
        Experience: {exp.get('role', '')} at {exp.get('company', '')}
        Duration: {exp.get('duration', '')}
        Location: {exp.get('location', '')}
        Description: {exp.get('description', '')}
        Responsibilities: {', '.join(exp.get('responsibilities', []))}
        Technologies: {', '.join(exp.get('technologies', []))}
        """
        chunks.append({
            "id": f"experience-{exp.get('id', 'unknown')}",
            "text": exp_text.strip(),
            "type": "experience",
            "metadata": {"section": "experience", "company": exp.get('company')}
        })
    
    # Education chunks
    for edu in data.get('education', []):
        edu_text = f"""
        Education: {edu.get('degree', '')} from {edu.get('institution', '')}
        Duration: {edu.get('duration', '')}
        Location: {edu.get('location', '')}
        Description: {edu.get('description', '')}
        Key Courses: {', '.join(edu.get('courses', []))}
        """
        chunks.append({
            "id": f"education-{edu.get('id', 'unknown')}",
            "text": edu_text.strip(),
            "type": "education",
            "metadata": {"section": "education", "institution": edu.get('institution')}
        })
    
    # Certifications chunks
    for cert in data.get('certifications', []):
        cert_text = f"""
        Certification: {cert.get('name', '')}
        Issuer: {cert.get('issuer', '')}
        Date: {cert.get('date', '')}
        URL: {cert.get('url', '')}
        """
        chunks.append({
            "id": f"certification-{cert.get('id', 'unknown')}",
            "text": cert_text.strip(),
            "type": "certification",
            "metadata": {"section": "certifications", "name": cert.get('name')}
        })
    
    # Contact chunk
    contact = data.get('contact', {})
    social = contact.get('social_links', {})
    contact_text = f"""
    Contact Information:
    Email: {data.get('profile', {}).get('email', '')}
    Availability: {contact.get('availability', '')}
    Use email for contact.
    GitHub: {social.get('github', '')}
    LinkedIn: {social.get('linkedin', '')}
    """
    chunks.append({
        "id": "contact",
        "text": contact_text.strip(),
        "type": "contact",
        "metadata": {"section": "contact"}
    })
    
    return chunks

def clean_metadata(metadata):
    """Remove None/null values from metadata - Pinecone doesn't accept them."""
    return {k: v for k, v in metadata.items() if v is not None}

def load_data():
    """Main ingestion function with CLIP embeddings."""
    print("Clearing existing vectors...")
    delete_all_vectors()
    
    print("Loading portfolio data from JSON...")
    data = load_portfolio_data()
    
    print("\n📝 Creating text chunks...")
    chunks = create_text_chunks(data)
    print(f"   Found {len(chunks)} text chunks")
    
    vectors = []
    
    # 1. Embed all text chunks
    print("\n🔤 Embedding text chunks with OpenCLIP...")
    for chunk in chunks:
        print(f"  Text: {chunk['id']}")
        try:
            embedding = get_text_embedding(chunk['text'])
            
            raw_metadata = {
                "text": chunk['text'],
                "type": chunk['type'],
                **chunk.get('metadata', {})
            }
            cleaned_metadata = clean_metadata(raw_metadata)
            
            vectors.append({
                "id": chunk['id'],
                "values": embedding,
                "metadata": cleaned_metadata
            })
        except Exception as e:
            print(f"  ERROR embedding text {chunk['id']}: {e}")
            continue
    
    # 2. Embed images (creates separate vectors in same space)
    print("\n🖼️ Embedding images with OpenCLIP...")
    for chunk in chunks:
        image_url = chunk.get('metadata', {}).get('image_url')
        if image_url and image_url.startswith('http'):
            print(f"  Image: {chunk['id']}")
            try:
                img_embedding = get_image_embedding(image_url)
                
                # Image vector gets same metadata as parent chunk
                raw_metadata = {
                    "text": f"[Image for {chunk.get('metadata', {}).get('title', chunk['id'])}]",
                    "type": f"{chunk['type']}_image",
                    "parent_id": chunk['id'],
                    **chunk.get('metadata', {})
                }
                cleaned_metadata = clean_metadata(raw_metadata)
                
                vectors.append({
                    "id": f"{chunk['id']}-image",
                    "values": img_embedding,
                    "metadata": cleaned_metadata
                })
            except Exception as e:
                print(f"  WARNING: Could not embed image for {chunk['id']}: {e}")
                continue
    
    if vectors:
        print(f"\n⬆️ Upserting {len(vectors)} vectors to Pinecone...")
        upsert_vectors(vectors)
        print("✅ Ingestion Complete!")
    else:
        print("❌ No vectors to upsert.")

if __name__ == "__main__":
    load_data()
