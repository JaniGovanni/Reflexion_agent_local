from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from typing import Union
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from output_schemas import CritiqueOutput, QualityAssessmentOutput
from langgraph.prebuilt import ToolNode
from langchain_community.tools.tavily_search import TavilySearchResults
from trustcall import create_extractor

def get_initial_writing_chain(llm: Union[ChatOllama, ChatGroq, ChatAnthropic]):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are expert researcher. Write a~250 word detailed answer to the users question."),
        ("human", "{question}")
    ])

    return prompt | llm

def get_critique_chain(llm: Union[ChatOllama, ChatGroq, ChatAnthropic]):
    instruction = """
    Review the generated content below and provide a detailed critique:

    {generation}

    Your critique should include:
    1. Identify any missing elements that would enhance the content.
    2. Highlight any parts that are unnecessary or redundant.

    Please follow the format instructions provided below:
    - "missing": List of elements missing from the content
    - "superfluous": List of superfluous elements in the content

    Example output format:
    {{
        "missing": ["item 1", "item 2"],
        "superfluous": ["item 1", "item 2"]
    }}
    """
    
    parser = PydanticOutputParser(pydantic_object=CritiqueOutput)
    
    critique_template = ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("user", "Please provide the critique in the specified format.") # important for llama3.2 3b
    ])

    return critique_template.partial(format_instructions=parser.get_format_instructions()) | llm | parser


def get_critique_chain_experimental(llm_json_mode: Union[ChatOllama, ChatGroq]):
    
    instruction = """
            Analyze the following generated content and provide a critique:

            {generation}

            Reflect on this content and provide:
            1. What is missing from this content?
            2. What is superfluous or unnecessary in this content?

            Respond in the following format:
            Missing: [Your critique of what's missing]
            Superfluous: [Your critique of what's superfluous]

            Example output format:
            {{
                "missing": ["item 1", "item 2"],
                "superfluous": ["item 1", "item 2"]
            }}
            """
    critique_template = ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("system", "Return JSON with two two keys, missing (critique of what is missing)"
                   "and superfluous (critique of what is superfluous).")])

    return critique_template | llm_json_mode

def get_web_search_chain_and_executor(llm: Union[ChatOllama, ChatGroq, ChatAnthropic]):
    tavily_search = TavilySearchResults(k=1)
    tool_executor = ToolNode([tavily_search])
    #tool_model = ChatGroq(model='llama3-groq-8b-8192-tool-use-preview').bind_tools([tavily_search])
    tool_model = llm.bind_tools([tavily_search])

    instruction = """
    Based on the following critique of a generated answer, improve the answer by searching for additional information. Do that by
    only performing maximum 3 searches:

    Original answer: {generation}
    Missing: {missing}
    Superfluous: {superfluous}
    """

    query_template = ChatPromptTemplate.from_messages([
        ("system", instruction),
    ])

    return query_template | tool_model, tool_executor


def get_rewrite_chain(llm: Union[ChatOllama, ChatGroq, ChatAnthropic]):
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert researcher tasked with improving an answer based on additional information. 
        Rewrite the Original answer, incorporating relevant new information and citing sources.
        Use footnote-style citations (e.g., [1], [2]) and include a references section at the end.
        Aim for a comprehensive, well-structured response of about 250-300 words."""),
        ("system", "Question: {question}"),
        ("system", "Original answer: {generation}"),
        ("system", "Additional information: {web_search_results}"),
        ("human", "Please rewrite the answer, incorporating the new information and adding citations.")
    ])

    return rewrite_prompt | llm


def get_quality_assessment_chain(llm: Union[ChatOllama, ChatGroq, ChatAnthropic]):
    instruction = """
    Perform a thorough analysis of the following answer to determine if it requires further improvements:

    Question: {question}
    Current Answer: {generation}

    Evaluate the answer based on these specific criteria:

    1. Completeness
       - Does it address all aspects of the question?
       - Are there any missing key concepts or important details?
       - Does it provide sufficient context and background information?

    2. Accuracy
       - Is all information factually correct?
       - Are claims properly supported with evidence or citations?
       - Are there any contradictions or logical inconsistencies?

    3. Clarity and Structure
       - Is the answer organized in a logical flow?
       - Are complex concepts explained in an accessible way?
       - Is the writing concise and free of unnecessary jargon?
       - Are transitions between ideas smooth and coherent?

    Be critical in your assessment. Even minor issues should be flagged for improvement.

    Example output format:
    {{
       "needs_improvement": true,
        "reasoning": "The answer requires improvement because it lacks detailed explanation of concept X."
    }}
    """
    
    parser = PydanticOutputParser(pydantic_object=QualityAssessmentOutput)
    
    quality_template = ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("user", "Please provide the assessment in the specified format.")
    ])

    return quality_template | llm | parser