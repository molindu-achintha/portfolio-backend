from groq import Groq
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

# Using Llama 3.3 70B - Current Flagship on Groq
MODEL_NAME = "llama-3.3-70b-versatile"

def generate_response(query: str, context: str, history: str = "") -> str:
    """
    Generate a response using Groq (Llama 3).
    Now includes session history for smarter suggestions.
    """
    if not settings.GROQ_API_KEY:
        return "Error: GROQ_API_KEY is missing in .env."

    try:
        client = Groq(api_key=settings.GROQ_API_KEY)
        
        system_prompt = (
            "You are a helpful and professional AI assistant for Molindu Achintha's portfolio. "
            "You answer questions based strictly on the provided context. "
            "If the answer is not in the context, politely say you don't know.\n\n"
            "IMPORTANT RESPONSE GUIDELINES:\n"
            "- Provide DETAILED and COMPREHENSIVE answers.\n"
            "- Include specific examples, technologies, and achievements from the context.\n"
            "- Use bullet points and structure your response clearly.\n"
            "- Explain the 'why' and 'how', not just the 'what'.\n"
            "- Make responses informative and thorough.\n\n"
            "SUGGESTION GUIDELINES:\n"
            "At the very end of your response, generate exactly 3 follow-up questions.\n"
            "- Base suggestions on the DATABASE CONTENT (projects, skills, experience available).\n"
            "- Avoid suggesting topics already discussed in the CONVERSATION HISTORY.\n"
            "- Make suggestions explore NEW areas of the portfolio not yet covered.\n"
            "- Keep suggestions short (under 10 words each).\n"
            "Format them exactly like this:\n"
            "<<SUGGESTIONS>>\n"
            "Question 1\n"
            "Question 2\n"
            "Question 3"
        )

        # Build user content with optional history
        user_content = f"Context:\n{context}\n\n"
        if history:
            user_content += f"Conversation History:\n{history}\n\n"
        user_content += f"Current Question:\n{query}"

        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            model=MODEL_NAME,
            temperature=0.6,
            max_tokens=2048,
        )

        return completion.choices[0].message.content

    except Exception as e:
        logger.error(f"Groq API Error: {e}")
        return f"Error generating response: {str(e)}"
