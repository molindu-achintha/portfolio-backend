import open_clip
import torch
import logging
import requests
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)

# Using ViT-B-32 with LAION-2B pretrained weights
# 512-dimensional embeddings, good balance of speed and quality
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

# Global model instances (loaded once)
_model = None
_preprocess = None
_tokenizer = None
_device = None

def _load_model():
    """Load OpenCLIP model (lazy initialization)."""
    global _model, _preprocess, _tokenizer, _device
    
    if _model is None:
        logger.info(f"Loading OpenCLIP model: {MODEL_NAME} ({PRETRAINED})")
        
        # Force CPU usage on Cloud Run to avoid BFloat16/MPS issues
        # Cloud Run environments often don't support BFloat16 well on CPUs
        _device = torch.device("cpu")
        logger.info(f"Using device: {_device} (Forced for Cloud Run compatibility)")
        
        # Load model with forced precision=fp32
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, 
            pretrained=PRETRAINED,
            precision='fp32',
            device=_device
        )
        _model.eval()
        
        _tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        logger.info("OpenCLIP model loaded successfully")
    
    return _model, _preprocess, _tokenizer, _device

def get_text_embedding(text: str) -> list:
    """
    Generate embedding for text using CLIP text encoder.
    Returns a list of 512 floats.
    """
    model, _, tokenizer, device = _load_model()
    
    try:
        # Disable autocast - force float32
        with torch.no_grad():
            text_tokens = tokenizer([text]).to(device)
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Ensure float32 output
            embedding = text_features[0].float().cpu().numpy().tolist()
            return [float(x) for x in embedding]
            
    except Exception as e:
        logger.error(f"Text embedding error: {e}")
        raise e

def get_image_embedding(image_url: str) -> list:
    """
    Download and embed an image from URL using CLIP image encoder.
    Returns a list of 512 floats.
    """
    model, preprocess, _, device = _load_model()
    
    try:
        # Download image
        logger.info(f"Downloading image: {image_url}")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Disable autocast - force float32
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Ensure float32 output
            embedding = image_features[0].float().cpu().numpy().tolist()
            return [float(x) for x in embedding]
            
    except Exception as e:
        logger.error(f"Image embedding error for {image_url}: {e}")
        raise e

def get_embedding(text: str) -> list:
    """
    Alias for get_text_embedding to maintain compatibility with existing code.
    """
    return get_text_embedding(text)
