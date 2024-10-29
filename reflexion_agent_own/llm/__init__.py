from langchain_ollama import ChatOllama
import os
import dotenv
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

dotenv.load_dotenv()
def get_ollama_llm():
    return ChatOllama(#model="phi3.5",
                      model="llama3.2",
                      temperature=0)

def get_groq_llm():
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        #model="llama-3.2-90b-text-preview",
        temperature=0,
        api_key=os.environ.get('GROQ_API_KEY')
    )
    return llm

def get_anthropic_llm():
    return ChatAnthropic(
        model="claude-3.5-sonnet-20240620",
        anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY'),
        temperature=0
    )