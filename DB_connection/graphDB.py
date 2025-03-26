import json
import os
import json
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.graphs import OntotextGraphDBGraph
from langchain.chains import OntotextGraphDBQAChain
from typing import Dict, TypedDict
from langgraph.graph import StateGraph


os.environ["GOOGLE_API_KEY"] = "AIzaSyB1j50tKq4yMm4vOQrqWb9yZ1p-G8ZP07Q"

# Initialize LLM
# llm = ChatOpenAI(model_name="gpt-4", temperature=0)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# take LLM from main.py

# Connect to Ontotext GraphDB
graph = OntotextGraphDBGraph(
    query_endpoint="http://LP148:7200/repositories/genai_graph_DB",
    query_ontology="CONSTRUCT {?s ?p ?o} WHERE {?s ?p ?o}"
)

# you can take query_endpoint through main.py 

# Define prompt template for query generation
template = """Convert this question into an optimized GraphDB query with appropriate filters: {query}"""
prompt = PromptTemplate(input_variables=["query"], template=template)

# Initialize query chain
chain = OntotextGraphDBQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    result_limit=100 
)


def define_workflow(question):
    workflow = StateGraph(GraphState)
    workflow.add_node("process_query", lambda x: {"result": chain.invoke(question)})
    # Define edges
    workflow.set_entry_point("process_query")
    workflow.set_finish_point("process_query")
    
    # Compile the graph
    app = workflow.compile()
    
    return app

def execute_query(question):
    """Executes a query on Ontotext GraphDB and handles large result sets."""
    try:
        workflow_app = define_workflow(question)
        result = workflow_app.invoke({"query": question})
        
        # Extract the query and the result from the response
        query = result.get("query", "")
        query_result = result.get("result", {}).get("result", "")
        
        if len(query_result) > 4000:
            return json.dumps({
                "query": query,
                "truncated_result": query_result[:4000] + "...",
                "note": "Result was truncated due to size. Please refine your query."
            }, indent=2)
        
        # Return the query along with the result
        return json.dumps({
            "query": query,
            "result": query_result
        }, indent=2)
    
    except Exception as e:
        return f"Error processing request: {str(e)}"



# Example Usage
if __name__ == "__main__":
    question = "count of transformers"
    result = execute_query(question)
    print(result)
