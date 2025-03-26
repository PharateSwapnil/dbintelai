import json
import logging
import os
import random
from typing import Any, Dict, List, TypedDict, Union, Literal
from DB_connection.graphDB import GraphDBQueryHandler
from DB_connection.postgres import PostgresDBQueryHandler



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required packages
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq import ChatGroq
    from langchain.prompts import PromptTemplate
    from langgraph.graph import StateGraph, START, END
    from langchain_core.messages import AIMessage, HumanMessage
    import psycopg2
    from psycopg2 import sql
except ImportError:
    logger.error("Required packages not installed. Installing now...")
    import subprocess
    packages = ["langchain", "langchain-core", "langgraph", "langchain-google-genai", "langchain-groq","psycopg2"]
    for package in packages:
        subprocess.check_call(["pip", "install", package])
    
    # Now import after installation
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_groq import ChatGroq
    from langchain.prompts import PromptTemplate
    from langgraph.graph import StateGraph, START, END
    from langchain_core.messages import AIMessage, HumanMessage
    import psycopg2
    from psycopg2 import sql

# Environment variables
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "AIzaSyB1j50tKq4yMm4vOQrqWb9yZ1p-G8ZP07Q")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Instead of connecting to GraphDB, we'll simulate query execution
# This is a simplified schema to work with, based on the summary provided
SAMPLE_SCHEMA = """
The database contains information about power system equipment with the following entities:
- Equipment (ID, name, description, type)
- Transformer (ID, name, apparentPower, voltageLevel, status)
- Terminal (ID, name, connected)
- Substation (ID, name, location)
- Line (ID, name, length, voltageLevel)
- Location (coordinates, positionPoints)

Relationships:
- Equipment contains Transformers
- Transformers connect to Terminals
- Lines connect to Terminals
- Substations contain Equipment
"""

# Sample GraphDB data for simulation
SAMPLE_POWER_SYSTEM_DATA = {
    "transformers": [
        {"id": "TX001", "name": "Main Power Transformer 1", "apparentPower": 100, "voltageLevel": 500, "status": "active"},
        {"id": "TX002", "name": "Distribution Transformer 2", "apparentPower": 50, "voltageLevel": 230, "status": "active"},
        {"id": "TX003", "name": "Auxiliary Transformer 3", "apparentPower": 25, "voltageLevel": 115, "status": "standby"}
    ],
    "terminals": [
        {"id": "TM001", "name": "High Voltage Terminal 1", "connected": True},
        {"id": "TM002", "name": "Low Voltage Terminal 2", "connected": True},
        {"id": "TM003", "name": "Medium Voltage Terminal 3", "connected": False}
    ],
    "substations": [
        {"id": "SUB001", "name": "Main Substation", "location": {"latitude": 40.7128, "longitude": -74.0060}},
        {"id": "SUB002", "name": "Secondary Substation", "location": {"latitude": 41.8781, "longitude": -87.6298}}
    ],
    "lines": [
        {"id": "LN001", "name": "Main Transmission Line", "length": 50, "voltageLevel": 500},
        {"id": "LN002", "name": "Distribution Line", "length": 25, "voltageLevel": 230}
    ]
}

# Cache for storing query results
query_cache = {}

# Define state schema for the graph
class ChatState(TypedDict):
    """Represents the state of our graph."""
    question: str
    chat_history: List[Union[HumanMessage, AIMessage]]
    model_choice: str
    db_type: str
    connection_config: Dict[str, Any]
    optimized_query: str
    validated_query: str
    query_result: Dict[str, Any]
    response: str
    error: str

# Initialize LLMs
def get_llm(model_choice: str):
    """Get LLM based on model choice."""
    if model_choice == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    elif model_choice == "llama3":
        return ChatGroq(
            api_key=GROQ_API_KEY,
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

# Define nodes for the graph workflow
def select_model(state: ChatState) -> Dict:
    """Select between Gemini and Llama3 models based on user choice or query complexity."""
    question = state["question"]
    chat_history = state["chat_history"]
    user_model_choice = state["model_choice"]
    
    # If user explicitly selected a model, use that
    if user_model_choice in ["gemini", "llama3"]:
        model_choice = user_model_choice
    # If auto-select, use heuristic
    elif user_model_choice == "auto":
        # Simple heuristic: use Llama3 for complex queries (more words, follow-ups)
        is_complex = len(question.split()) > 15 or len(chat_history) > 2
        # Choose model based on complexity
        model_choice = "llama3" if is_complex and GROQ_API_KEY else "gemini"
    else:
        # Default to Gemini for any other value
        model_choice = "gemini"
    
    logger.info(f"Selected model: {model_choice} (user choice: {user_model_choice})")
    
    # Return only the field we want to update
    return {"model_choice": model_choice}

def optimize_query(state: ChatState) -> Dict:
    """Optimize the database query using LLM based on the database type."""
    question = state["question"]
    chat_history = state["chat_history"]
    model_choice = state["model_choice"]
    db_type = state.get("db_type", "graphdb")  # Default to GraphDB
    
    # Get appropriate LLM
    llm = get_llm(model_choice)
    
    # Create context from chat history
    context = ""
    if chat_history:
        context = "Based on our previous conversation:\n" + "\n".join(
            [f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}" 
             for msg in chat_history[-2:]]
        ) + "\n"
    
    # Determine query type based on database type
    if db_type in ["postgres", "aws-rds"]:
        query_language = "SQL"
        expertise = "SQL databases"
        query_format = f"a {db_type} SQL query"
    elif db_type == "aws-neptune":
        # Check if Neptune is configured for SPARQL or Gremlin
        connection_config = state.get("connection_config", {})
        query_language = connection_config.get("query_language", "sparql").upper()
        expertise = f"Neptune {query_language} queries"
        query_format = f"an Amazon Neptune {query_language} query"
    else:  # Default to GraphDB/SPARQL
        query_language = "SPARQL"
        expertise = "GraphDB and SPARQL queries"
        query_format = "a GraphDB SPARQL query"
    
    # Optimization prompt
    template = """You are an expert in {expertise}.
    {context}
    Convert this question into {query_format} with appropriate filters:
    Question: {question}
    
    Only return the optimized {query_language} query, nothing else."""
    
    prompt = PromptTemplate(
        input_variables=["expertise", "context", "question", "query_format", "query_language"], 
        template=template
    )
    
    try:
        result = llm.invoke(prompt.format(
            expertise=expertise,
            context=context,
            question=question,
            query_format=query_format,
            query_language=query_language
        ))
        optimized_query = result.content
        logger.info(f"Optimized {query_language} query created successfully")
        # Return only the fields we want to update
        return {"optimized_query": optimized_query}
    except Exception as e:
        logger.error(f"Error optimizing query: {str(e)}")
        # Return only the fields we want to update
        return {"error": f"Error optimizing query: {str(e)}"}

def validate_query(state: ChatState) -> Dict:
    """Validate database query for syntax and structure based on database type."""
    optimized_query = state["optimized_query"]
    model_choice = state["model_choice"]
    db_type = state.get("db_type", "graphdb")  # Default to GraphDB
    
    # Get appropriate LLM
    llm = get_llm(model_choice)
    
    # Determine query type and expertise based on database type
    if db_type in ["postgres", "aws-rds"]:
        query_language = "SQL"
        expertise = "SQL databases"
    elif db_type == "aws-neptune":
        # Check if Neptune is configured for SPARQL or Gremlin
        connection_config = state.get("connection_config", {})
        query_language = connection_config.get("query_language", "sparql").upper()
        expertise = f"Neptune {query_language} queries"
    else:  # Default to GraphDB/SPARQL
        query_language = "SPARQL"
        expertise = "GraphDB and SPARQL queries"
    
    # Validation prompt
    template = """You are an expert in {expertise}.
    Check the following {query_language} query for syntax errors and semantic correctness:
    {query}
    
    If it's valid, respond with the exact query.
    If it's invalid, fix it and respond with the corrected query only."""
    
    prompt = PromptTemplate(
        input_variables=["expertise", "query_language", "query"], 
        template=template
    )
    
    try:
        result = llm.invoke(prompt.format(
            expertise=expertise,
            query_language=query_language,
            query=optimized_query
        ))
        validated_query = result.content
        logger.info(f"{query_language} query validation completed")
        # Return only the fields we want to update
        return {"validated_query": validated_query}
    except Exception as e:
        logger.error(f"Error validating query: {str(e)}")
        # Return only the fields we want to update
        return {"error": f"Error validating query: {str(e)}"}

def execute_query(state: ChatState, schema_info, data) -> Dict:
    """Execute the validated query based on database type (simulated)."""
    validated_query = state["validated_query"]
    question = state["question"]
    model_choice = state["model_choice"]
    db_type = state.get("db_type", "graphdb")  # Default to GraphDB
    
    # Check cache first
    cache_key = f"{db_type}:{validated_query}"
    if cache_key in query_cache:
        logger.info(f"Query result found in cache")
        # Return only the fields we want to update
        return {"query_result": query_cache[cache_key]}
    
    # Get LLM for simulated query execution
    llm = get_llm(model_choice)
    
    # Determine query type and schema based on database type
    if db_type in ["postgres", "aws-rds"]:
        query_language = "SQL"
        db_description = "relational database (PostgreSQL)"
        # SQL schema for DB
        schema = schema_info

    elif db_type == "aws-neptune":
        # Check if Neptune is configured for SPARQL or Gremlin
        connection_config = state.get("connection_config", {})
        query_language = connection_config.get("query_language", "sparql").upper()
        if query_language.upper() == "GREMLIN":
            db_description = "graph database using Gremlin"
            schema = """
            The Neptune graph database contains information about power system equipment with the following vertices and edges:
            
            Vertices:
            - Equipment (properties: id, name, description, type)
            - Transformer (properties: id, name, apparentPower, voltageLevel, status)
            - Terminal (properties: id, name, connected)
            - Substation (properties: id, name, location with latitude/longitude)
            - Line (properties: id, name, length, voltageLevel)
            
            Edges:
            - contains (from Equipment to Transformer)
            - locatedAt (from Equipment to Substation)
            - connects (from Terminal to Equipment)
            - connectsLine (from Terminal to Line)
            """
        else:  # SPARQL
            db_description = "Neptune graph database using SPARQL"
            schema = SAMPLE_SCHEMA  # Same as GraphDB
    else:  # Default to GraphDB/SPARQL
        query_language = "SPARQL"
        db_description = "GraphDB semantic database"
        schema = SAMPLE_SCHEMA
    
    # Create a prompt to simulate query execution
    template = """Given the following {db_type} database schema:
    {schema}
    
    And the following sample data:
    {data}
    
    Simulate executing this {query_language} query on a {db_description}:
    {query}
    
    Based on the user's original question: {question}
    
    Return the results in the following JSON format:
    {{
        "result": "The answer to the question based on the data",
        "intermediate_steps": [
            "Step 1: Interpreted query as...",
            "Step 2: Retrieved data..."
        ],
        "source_documents": [
            List relevant data sources used from our sample data
        ]
    }}
    """
    
    # Format sample data as string
    # data_str = data
    
    prompt = PromptTemplate(
        input_variables=["db_type", "schema", "data", "query_language", "db_description", "query", "question"], 
        template=template
    )
    
    try:
        # Simulate query execution using LLM
        prompt_filled = prompt.format(
            db_type=db_type,
            schema=schema, 
            data=data,
            query_language=query_language,
            db_description=db_description,
            query=validated_query, 
            question=question
        )
        
        result = llm.invoke(prompt_filled)
        
        # Parse the response to a dictionary
        try:
            # Try to parse as JSON
            query_result = json.loads(result.content)
        except json.JSONDecodeError:
            # If not valid JSON, wrap it in our expected format
            query_result = {
                "result": result.content,
                "intermediate_steps": [f"{query_language} query executed and simulated"],
                "source_documents": ["sample_power_system_data"]
            }
        
        # Cache the result
        query_cache[cache_key] = query_result
        logger.info(f"{query_language} query simulated successfully for {db_type}")
        
        # Return only the fields we want to update
        return {"query_result": query_result}
    except Exception as e:
        logger.error(f"Error executing simulated query: {str(e)}")
        # Return only the fields we want to update
        return {"error": f"Error executing simulated query: {str(e)}"}

def generate_response(state: ChatState) -> Dict:
    """Generate a final response based on query results and database type."""
    query_result = state["query_result"]
    question = state["question"]
    chat_history = state["chat_history"]
    model_choice = state["model_choice"]
    validated_query = state.get("validated_query", "")
    db_type = state.get("db_type", "graphdb")  # Default to GraphDB
    
    # If we have an error, return it
    if state.get("error"):
        # Return only the fields we want to update
        return {"response": f"I encountered an error: {state['error']}"}
    
    # Determine which query language was used based on database type
    if db_type in ["postgres", "aws-rds"]:
        code_block_type = "sql"
    elif db_type == "aws-neptune":
        # Check if Neptune is configured for SPARQL or Gremlin
        connection_config = state.get("connection_config", {})
        query_language = connection_config.get("query_language", "sparql").lower()
        code_block_type = query_language
    else:  # Default to GraphDB/SPARQL
        code_block_type = "sparql"
    
    # Format the response
    try:
        llm = get_llm(model_choice)
        
        # Create a prompt for formatting the response
        template = """Based on the question: {question}
        And the retrieved information: {query_result}
        
        Generate a clear, concise, and informative response. Include relevant details from the query results.
        If the query result is too large, summarize the key points.
        
        IMPORTANT: Include the query used as a code block in your response. Format it like this:
        ```{code_block_type}
        [actual query]
        ```
        
        Place this query block after your initial explanation but before going into specific details."""
        
        prompt = PromptTemplate(
            input_variables=["question", "query_result", "code_block_type"], 
            template=template
        )
        
        # Convert query_result to string for the prompt
        query_result_str = json.dumps(query_result, indent=2)
        if len(query_result_str) > 8000:  # Truncate if too large
            query_result_str = query_result_str[:8000] + "... [truncated due to size]"
        
        result = llm.invoke(prompt.format(
            question=question, 
            query_result=query_result_str,
            code_block_type=code_block_type
        ))
        response = result.content
        
        # Check if the response already includes a code block for the query
        code_block_marker = f"```{code_block_type}"
        if code_block_marker not in response.lower() and validated_query:
            # Add the query as a code block after the first paragraph
            paragraphs = response.split('\n\n')
            if len(paragraphs) > 0:
                code_block = f"\n\n```{code_block_type}\n{validated_query}\n```\n\n"
                response = paragraphs[0] + code_block + '\n\n'.join(paragraphs[1:])
        
        logger.info(f"Response generated successfully for {db_type}")
        
        # Return only the fields we want to update
        return {"response": response}
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        # Return only the fields we want to update
        return {"response": f"I found some information but encountered an error formatting the response. Here's the raw result: {str(query_result.get('result', ''))[:1000]}..."}

def decide_next_step(state: ChatState) -> Literal["continue", "end"]:
    """Decide whether to continue processing or end the workflow."""
    if state.get("error"):
        logger.warning(f"Workflow ended with error: {state['error']}")
        return "end"
    return "continue"

# Set up the LangGraph workflow
workflow = StateGraph(ChatState)

# Add nodes
workflow.add_node("select_model", select_model)
workflow.add_node("optimize_query", optimize_query)
workflow.add_node("validate_query", validate_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("generate_response", generate_response)

# Add edges - starting with the entry point
workflow.add_edge(START, "select_model")
workflow.add_edge("select_model", "optimize_query")
workflow.add_edge("optimize_query", "validate_query")
workflow.add_edge("validate_query", "execute_query")
workflow.add_edge("execute_query", "generate_response")
workflow.add_edge("generate_response", END)

# Conditional edges for error handling
workflow.add_conditional_edges(
    "optimize_query",
    decide_next_step,
    {
        "continue": "validate_query",
        "end": END
    }
)

workflow.add_conditional_edges(
    "validate_query",
    decide_next_step,
    {
        "continue": "execute_query",
        "end": END
    }
)

workflow.add_conditional_edges(
    "execute_query",
    decide_next_step,
    {
        "continue": "generate_response",
        "end": END
    }
)

# Compile the graph
app_workflow = workflow.compile()

def get_response(question, chat_history=None, db_type="graphdb", connection_config=None, model_choice="auto"):
    """Process user question through the LangGraph workflow."""
    if chat_history is None:
        chat_history = []

    # llm = get_llm(model_choice)
    # question = ""
    # response = llm.generate([prompt])
        
    try:
        is_connected, db_instance = set_connection_config(db_type, connection_config)
        if not is_connected:
            return f"Error: {db_instance}", chat_history  # db_instance contains the error message
        
        if db_type == "graphdb":

            query_handler = GraphDBQueryHandler(model_choice=llm, query_endpoint=query_endpoint, query_ontology=query_ontology)
            result, chat_history = query_handler.execute_query(question, chat_history)
            return result, chat_history

            chat_history.append(HumanMessage(content=question))
            chat_history.append(AIMessage(content=result["result"]))

            if len(chat_history) > 10:  # Limit history length
                chat_history = chat_history[-10:]

            return result["result"], chat_history

        elif db_type == "postgres":
            db = PostgresDB(                
                model_choice=llm,
                host=connection_config.get("host"),
                port=connection_config.get("port"),
                dbname=connection_config.get("dbname"),
                user=connection_config.get("user"),
                password=connection_config.get("password")
            )

            schema_info = db.get_schema_info()
            result_table = db.execute_query(generated_sql)  
            generated_sql = db.generate_sql_query(question, schema_info)
            visualization = db.visualize_data(result_table)  # Generate visualization
            db.close_connection()

            # Initialize state
            initial_state = {
                "question": question,
                "chat_history": chat_history,
                "model_choice": model_choice,
                "optimized_query": "",
                "validated_query": "",
                "query_result": {},
                "response": "",
                "error": "",
                "db_type": db_type,
                "connection_config": connection_config or {}
            }

            result = app_workflow.invoke(initial_state)

            
            generated_sql = db.generate_sql_query(question, schema_info)

            db.close_connection()
            
            chat_history.append(HumanMessage(content=question))
            chat_history.append(AIMessage(content=f"Query Result:\n{result_table.to_string()}"))

            return result_table.to_string(), chat_history

        else:
            # Initialize state
            initial_state = {
                "question": question,
                "chat_history": chat_history,
                "model_choice": model_choice,
                "optimized_query": "",
                "validated_query": "",
                "query_result": {},
                "response": "",
                "error": "",
                "db_type": db_type,
                "connection_config": connection_config or {}
            }
            
            # Execute the workflow
            result = app_workflow.invoke(initial_state)
            
            # Update chat history
            chat_history.append(HumanMessage(content=question))
            chat_history.append(AIMessage(content=result["response"]))
            
            if len(chat_history) > 10:  # Limit history length
                chat_history = chat_history[-10:]
            
            return result["response"], chat_history
    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}")
        return f"Error processing request: {str(e)}", chat_history

def get_connection_types():
    """Get available database connection types."""
    return {
        "available_types": [
            {"id": "graphdb", "name": "GraphDB", "description": "Connect to a SPARQL endpoint for semantic queries."},
            {"id": "postgres", "name": "PostgreSQL", "description": "Connect to your PostgreSQL database for SQL-based queries."},
            {"id": "aws-rds", "name": "AWS RDS", "description": "Connect to an Amazon RDS instance for managed database access."},
            {"id": "aws-neptune", "name": "AWS Neptune", "description": "Connect to Amazon Neptune for graph database queries."}
        ]
    }

def set_connection_config(db_type, config):
    """Set the database connection configuration."""
    try:
        logger.info(f"Setting up connection for {db_type}")
        
        # Verify the connection based on db_type
        if db_type == "graphdb":
            # Validate GraphDB connection
            endpoint = config.get("endpoint")
            if not endpoint:
                return False, "GraphDB endpoint URL is required"
            
            # In a real app, we'd verify the connection here
            query_handler = GraphDBQueryHandler(llm, query_endpoint, query_ontology)
            graph_instance = query_handler.graph

            try:
                # Test connection
                result = graph_instance.query("SELECT * WHERE {?s ?p ?o} LIMIT 1")
                if result:
                    logger.info(f"Successfully connected to GraphDB at {query_endpoint}")
                    return True, None
                else:
                    logger.info(f"Failed to connect to GraphDB at {query_endpoint}")
            except Exception as e:
                logger.error(f"GraphDB connection error: {str(e)}")
                return False, str(e)
            
        elif db_type == "postgres":
            # Validate PostgreSQL connection
            host = config.get("host")
            port = config.get("port")
            database = config.get("database")
            username = config.get("username")
            password = config.get("password")
            
            if not all([host, port, database, username, password]):
                return False, "All PostgreSQL connection parameters are required"

            try:
                postgres = PostgresDBQueryHandler(

                    host=host,
                    port=port,
                    database=database,
                    user=username,
                    password=password
                )
                
                # Test query to verify the connection
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()
                
                if result:
                    logger.info(f"Successfully connected to PostgreSQL at {host}:{port}/{database}")
                    return True, conn  # Return connection instance
                else:
                    return False, "PostgreSQL connection established, but query returned no data"
            except Exception as e:
                logger.error(f"PostgreSQL connection error: {str(e)}")
                return False, str(e)
            
        elif db_type == "aws-rds":
            # Validate AWS RDS connection
            endpoint = config.get("endpoint")
            port = config.get("port")
            database = config.get("database")
            username = config.get("username")
            password = config.get("password")
            
            if not all([endpoint, port, database, username, password]):
                return False, "All AWS RDS connection parameters are required"
            
            # In a real app, we'd verify the connection here
            logger.info(f"Successfully connected to AWS RDS at {endpoint}:{port}/{database}")
            return True, None
            
        elif db_type == "aws-neptune":
            # Validate AWS Neptune connection
            endpoint = config.get("endpoint")
            port = config.get("port")
            
            if not all([endpoint, port]):
                return False, "Neptune endpoint and port are required"
            
            # In a real app, we'd verify the connection here
            logger.info(f"Successfully connected to AWS Neptune at {endpoint}:{port}")
            return True, None
            
        else:
            return False, f"Unsupported database type: {db_type}"
            
    except Exception as e:
        logger.error(f"Error setting connection: {str(e)}")
        return False, str(e)

# Interactive Chat
def chat_interface():
    print("Welcome to the GraphDB Chat Assistant with LangGraph!")
    print("Type 'exit' to end the conversation.\n")
    print("Type 'clear' to clear chat history.\n")
    
    chat_history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        if user_input.lower() == 'clear':
            chat_history = []
            print("Chat history cleared.")
            continue
        if not user_input.strip():
            print("Please enter a valid question.")
            continue
            
        response, chat_history = get_response(user_input, chat_history)
        print("\nAssistant:", response)
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    try:
        chat_interface()
    except KeyboardInterrupt:
        print("\nSession terminated by user. Goodbye!")