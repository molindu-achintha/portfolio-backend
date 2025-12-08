from groq import Groq
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

# Using Llama 3.3 70B - Current Flagship on Groq
MODEL_NAME = "llama-3.3-70b-versatile"

def generate_response(query: str, context: str) -> str:
    """
    Generate a response using Groq (Llama 3).
    """
    if not settings.GROQ_API_KEY:
        return "Error: GROQ_API_KEY is missing in .env."

    try:
        client = Groq(api_key=settings.GROQ_API_KEY)
        
        system_prompt = (
            "You are a helpful and professional AI assistant for Molindu Achintha's portfolio. "
            "You answer questions based strictly on the provided context. "
            "If the answer is not in the context, politely say you don't know.\n\n"
            "IMPORTANT: At the very end of your response, strictly following the main answer, "
            "generate exactly 3 short, relevant follow-up questions that the user might want to ask next based on the context. "
            "Format them exactly like this:\n"
            "<<SUGGESTIONS>>\n"
            "Question 1\n"
            "Question 2\n"
            "Question 3"
        )

        user_content = f"Context:\n{context}\n\nQuestion:\n{query}"

        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            model=MODEL_NAME,
            temperature=0.5,
            max_tokens=1024,
        )

        return completion.choices[0].message.content

    except Exception as e:
        logger.error(f"Groq API Error: {e}")
        return f"Error generating response: {str(e)}"
