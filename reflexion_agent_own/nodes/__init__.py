from state import GraphState
from langchain_core.messages import AIMessage, HumanMessage
import json
from typing import Dict, Any
from chains import (get_initial_writing_chain,
                    get_critique_chain,
                    get_web_search_chain_and_executor,
                    get_rewrite_chain,
                    get_quality_assessment_chain)

def initial_node(state: GraphState) -> Dict[str, Any]:
    chain = get_initial_writing_chain(state["llm"])
    
    state["messages"] = [HumanMessage(content=state["question"])]
    result = chain.invoke(state["question"])
    
    
    return {
        "messages": state["messages"] + [AIMessage(content=result.content)],
        "generation": result.content
    }



def critique_node(state: GraphState) -> Dict[str, Any]:
    critique_chain = get_critique_chain(state["llm"])
    
    critique = critique_chain.invoke({"generation" : state["generation"]})
    
    reflection = critique.dict()
    
    return {
        "messages": state["messages"] + [AIMessage(content=json.dumps(reflection))],
        "reflection": reflection
    }



def web_search_query_node(state: GraphState) -> Dict[str, Any]:       
    query_chain, tool_executor = get_web_search_chain_and_executor(state["llm"])
    
    result = query_chain.invoke({
        "generation": state["generation"],
        "missing": state["reflection"]["missing"],
        "superfluous": state["reflection"]["superfluous"]
    })
    search_result = tool_executor.invoke([result])
    
   
    web_search = [res for message in search_result for res in json.loads(message.content)]

    return {
        "messages": state["messages"] + search_result,
        "web_search": web_search
    }


def rewrite_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    generation = state["generation"]
    web_search_results = state["web_search"]
    loop_step = state["loop_step"]
    
    rewrite_chain = get_rewrite_chain(state["llm"])
    
    result = rewrite_chain.invoke({
        "question": question,
        "generation": generation,
        "web_search_results": json.dumps(web_search_results, indent=2)
    })

    generation = result.content
    loop_step += 1
    
    return {
        "messages": state["messages"] + [AIMessage(content=result.content)],
        "generation": generation,
        "loop_step": loop_step
    }

def quality_assessment_node(state: GraphState) -> Dict[str, Any]:
    question = state["question"]
    generation = state["generation"]
    
    quality_chain = get_quality_assessment_chain(state["llm"])
    
    result = quality_chain.invoke({
        "question": question,
        "generation": generation,
    })
    
    return {
        "messages": state["messages"] + [AIMessage(content=json.dumps(result.dict()))],
        "needs_improvement": result.needs_improvement
    }