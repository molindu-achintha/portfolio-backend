import requests
import json
import logging
import re
from app.core.config import settings

logger = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def generate_response(query: str, context: str, history: str = "") -> str:
    """Generate a response using OpenRouter (Mistral Large)."""
    if not settings.OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY is missing in .env."

    try:
        prompt = f"""You are Molindu Achintha, a Computer Science student from Sri Lanka.
Answer questions about your portfolio in FIRST PERSON (use "I", "my", "me").

RESPONSE STRUCTURE:
1. Think through the question step by step
2. Provide a detailed, comprehensive answer

FORMAT RULES:
- Use ### for section headers
- Use **bold** SPARINGLY - only for key project names and technologies
- Use bullet points for lists
- Be extremely detailed and thorough

IMPORTANT - END WITH SUGGESTIONS:
At the very end, add exactly 3 follow-up questions:

<<SUGGESTIONS>>
Question about a specific project?
Question about technologies?
Question about experience?

PORTFOLIO CONTEXT:
{context}

USER QUESTION: {query}

Think through this carefully, then provide your detailed answer:"""

        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://molindu-achintha.github.io",
            "X-Title": "Molindu Portfolio AI",
        }

        data = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(
            url=OPENROUTER_URL,
            headers=headers,
            data=json.dumps(data),
            timeout=60
        )

        if response.status_code != 200:
            logger.error(f"OpenRouter Error: {response.status_code} - {response.text}")
            return f"Error: API returned status {response.status_code}"

        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            output = result['choices'][0].get('message', {}).get('content', '')
        else:
            output = str(result)
        
        # Format thinking blocks
        def format_thinking(match):
            thinking_content = match.group(1).strip()
            return f"""### 💭 Thinking Process

> {thinking_content.replace(chr(10), chr(10) + '> ')}

---

### 📝 Response

"""
        
        output = re.sub(r'<think>(.*?)</think>', format_thinking, output, flags=re.DOTALL)
        output = output.strip()
        
        return output

    except requests.exceptions.Timeout:
        logger.error("OpenRouter request timed out")
        return "Error: Request timed out. Please try again."
    except Exception as e:
        logger.error(f"OpenRouter API Error: {e}")
        return f"Error generating response: {str(e)}"
