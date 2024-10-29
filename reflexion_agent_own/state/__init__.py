from typing import List, Union
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

class GraphState(TypedDict):
    question: str  # User question
    generation: str  # LLM generation
    max_retries: int  # Max number of retries for answer generation
    loop_step: int
    llm: Union[ChatOllama, ChatGroq, ChatAnthropic]
    reflection: dict = Field(default_factory=dict)
    web_search: List[dict] = Field(default_factory=list)  # informations provided by web search 
    messages: List[Union[HumanMessage, AIMessage, ToolMessage]] = Field(default_factory=list)
    needs_improvement: bool