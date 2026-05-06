from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
import os

USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"

def get_llm():
    if USE_GROQ:
        return ChatGroq(
            model="llama3-70b-8192",
            temperature=0
        )
    else:
        return ChatOllama(
            model="llama3",
            temperature=0
        )