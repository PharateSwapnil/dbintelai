import psycopg2
import pandas as pd
import sweetviz as sv

def get_schema_info(conn):
    """Fetch schema details for all schemas in the database."""
    cur = conn.cursor()
    query = """
    SELECT table_schema, table_name, column_name, data_type
    FROM information_schema.columns;
    """
    cur.execute(query)
    schema_info = cur.fetchall()
    organized_schema = {}
    
    for schema, table, column, dtype in schema_info:
        if schema not in organized_schema:
            organized_schema[schema] = {}
        if table not in organized_schema[schema]:
            organized_schema[schema][table] = {}
        
        organized_schema[schema][table][column] = dtype
    
    cur.close()
    return organized_schema

def generate_sql_query(prompt, schema_info):
    """Ask LLM to generate SQL based on schema and user question."""
    # Create a structured schema string to pass to the LLM
    schema_text = ""
    
    for schema, tables in schema_info.items():
        schema_text += f"Schema: {schema}\n"
        for table, columns in tables.items():
            columns_text = ", ".join([f"{col}: {dtype}" for col, dtype in columns.items()])
            schema_text += f"Table: {table}\nColumns: {{{columns_text}}}\n"
    
    # Construct the prompt for the LLM
    full_prompt = f"Database Schema Information:\n{schema_text}\n\nUser Query: {prompt}\nGenerate an SQL query:"
    
    # Replace with your LLM code (e.g., OpenAI)
    response = "Generated SQL Query: SELECT * FROM customers;"  # Replace with actual LLM API call
    return response  # Here, you would return the LLM's response

def execute_query(conn, query):
    """Run the generated SQL query on PostgreSQL."""
    cur = conn.cursor()
    cur.execute(query)
    results = cur.fetchall()
    cur.close()
    return results

def visualize_data(data):
    """Visualize the data using sweetviz"""
    df = pd.DataFrame(data)
    report = sv.analyze(df)
    report.show_html('sweetviz_report.html')

# Connect to PostgreSQL
conn = psycopg2.connect(host="0.0.0.0", port=5000, dbname="mydatabase", user="myuser", password="mypassword")
# for this you can take credentials from main.py

# Get schema info
schema_info = get_schema_info(conn)

# Generate SQL using LLM
user_prompt = "Get all customers from the public schema."
generated_sql = generate_sql_query(user_prompt, schema_info)

# Execute SQL
query_result = execute_query(conn, generated_sql)

# Visualize data
visualize_data(query_result)

## create a option for visualization in main.py

conn.close()
    
# In this script, we first connect to a PostgreSQL database using the  psycopg2  library. We then define a function  get_schema_info  to fetch schema details for all schemas in the database. Next, we define a function  generate_sql_query  to ask the large language model (LLM) to generate SQL based on the schema and user question. 
# The  execute_query  function runs the generated SQL query on PostgreSQL. Finally, we define a function  visualize_data  to visualize the data using the  sweetviz  library. 
# We then connect to the PostgreSQL database, get the schema information, generate SQL using the LLM, execute the SQL query, and visualize the data. 
