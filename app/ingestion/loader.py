"""
Portfolio Data Ingestion Script (Groq + Pinecone)
Loads portfolio data from JSON and creates embeddings for RAG retrieval.
Images are indexed by their metadata (Project Title/Description) for retrieval.
"""
import time
import uuid
import re
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from app.services import embedding_service
from app.services.vector_db import upsert_vectors, delete_all_vectors

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
        
        # Append documents if present so LLM knows about them
        if 'documents' in project:
            docs_str = "\n".join([f"- {d['name']} ({d['type']}): {d['url']}" for d in project['documents']])
            project_text += f"\nDocuments:\n{docs_str}\n"
        
        # Only PROJECTS have image_url in metadata
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
                "image_url": project.get('image'),  # Only projects have images
                "video_url": project.get('video')   # Only projects have videos
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
    """Main ingestion function."""
    print("Clearing existing vectors...")
    delete_all_vectors()
    
    print("Loading portfolio data from JSON...")
    data = load_portfolio_data()
    
    print("\n📝 Creating chunks...")
    chunks = create_text_chunks(data)
    print(f"   Found {len(chunks)} chunks")
    
    vectors = []
    for chunk in chunks:
        print(f"  Embedding: {chunk['id']}")
        try:
            # Generic embedding service (HF BGE)
            embedding = embedding_service.get_embedding(chunk['text'])
            time.sleep(0.5) 
            
            # Clean metadata
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
            print(f"  ERROR embedding {chunk['id']}: {e}")
            continue
    
    if vectors:
        print(f"\nUpserting {len(vectors)} vectors to Pinecone...")
        upsert_vectors(vectors)
        print("✅ Ingestion Complete!")
    else:
        print("❌ No vectors to upsert.")

if __name__ == "__main__":
    load_data()
