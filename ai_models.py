import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

def get_llm(model_choice: str):
    """Get LLM based on model choice."""
    if model_choice == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            google_api_key = os.getenv('GOOGLE_API_KEY')
        )
    elif model_choice == "llama3":
        return ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'),
            model_name="llama3-70b-8192",
            temperature=0.1,
            streaming=True
        )
    else:
        # Default to Gemini
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
  
