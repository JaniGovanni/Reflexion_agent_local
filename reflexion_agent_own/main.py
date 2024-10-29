from langgraph.graph import END, StateGraph
from nodes import initial_node, critique_node, web_search_query_node, rewrite_node, quality_assessment_node
from state import GraphState
from llm import get_groq_llm, get_ollama_llm
import os
import argparse

MAX_ITERATIONS = 2
builder = StateGraph(GraphState)

# Add nodes
builder.add_node("INITIAL_WRITER", initial_node)
builder.add_node("CRITIQUE_WRITER", critique_node)
builder.add_node("RESEARCHER", web_search_query_node)
builder.add_node("REWRITER", rewrite_node)
builder.add_node("QUALITY_ASSESSOR", quality_assessment_node)
# Add edges
builder.add_edge("INITIAL_WRITER", "CRITIQUE_WRITER")
builder.add_edge("CRITIQUE_WRITER", "RESEARCHER")
builder.add_edge("RESEARCHER", "REWRITER")
builder.add_edge("REWRITER", "QUALITY_ASSESSOR")

def event_loop(state: GraphState) -> str:
    # to limit max iterations
    if state["loop_step"] >= MAX_ITERATIONS:
        return END
    
    # if the answer needs improvement, go back to critique writer
    if state["needs_improvement"]:
        return "CRITIQUE_WRITER"
    else:
        return END

builder.add_conditional_edges(
    "QUALITY_ASSESSOR",
    event_loop,
    {
        "CRITIQUE_WRITER": "CRITIQUE_WRITER",
        END: END
    }
)
builder.set_entry_point("INITIAL_WRITER")

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")



def main():
    
    question = "What differs the contextual retrieval method invented by anthropic from a naive RAG system?"

    llm = get_ollama_llm()  # get_groq_llm() | get_anthropic_llm()

    initial_state = GraphState(
        question=question,
        max_retries=2,
        loop_step=0,
        llm=llm,
    )

    res = graph.invoke(initial_state)
    
    # Print the final answer
    print("Final Answer:")
    print(res["generation"])
    
    # Write the final answer to a text file
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "final_answer.md")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Question: {initial_state['question']}\n\n")
        f.write("Final Answer:\n")
        f.write(res["generation"])
    
    print(f"\nFinal answer has been written to: {output_file}")

if __name__ == "__main__":
    main()