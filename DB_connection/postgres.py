
import os
import json
from dotenv import load_dotenv
import psycopg2
import pandas as pd
import sweetviz as sv
import pandas as pd
import sweetviz as sv
# from ai_models import get_llm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
# from langchain.llms import OpenAI

# Load environment variables from the .env file
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
    
class PostgresDBQueryHandler:
    def __init__(self, host, port, dbname, user, password):
        """Initialize the PostgreSQL connection."""
        self.conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user, password=password
        )
        self.cursor = self.conn.cursor()
      
    def get_schema_info(self):
        """Fetch schema details for all schemas in the database."""

        query = """
        SELECT table_schema, table_name, column_name, data_type
        FROM information_schema.columns;
        """
        self.cursor.execute(query)
        schema_info = self.cursor.fetchall()
        organized_schema = {}
        
        for schema, table, column, dtype in schema_info:
            if schema not in organized_schema:
                organized_schema[schema] = {}
            if table not in organized_schema[schema]:
                organized_schema[schema][table] = {}
            
            organized_schema[schema][table][column] = dtype 

        return organized_schema
    
    def generate_sql_query(self, model_choice, prompt, schema_info):
        """Generate accurate SQL queries using structured schema info."""
        # # Initialize schema_text variable
        schema_text = ""
        # Check if 'public' schema exists in schema_info
        if 'public' in schema_info:
            # Iterate only over the 'public' schema
            schema_text += f"Schema: public\n"
            public_schema = schema_info['public']
            # Iterate over the tables in the 'public' schema
            for table, columns in public_schema.items():
                columns_text = ", ".join([f"{col}: {dtype}" for col, dtype in columns.items()])
                schema_text += f"Table: {table}\nColumns: {{{columns_text}}}\n"

        llm = get_llm(model_choice)

        full_prompt = f"""
        Analyze the following user prompt and generate a valid SQL query optimized for the given database schema.
        
        Database Schema Information:
        {schema_text}
        
        User Query: {prompt}
        
        Instructions:
        - dont generate CREATE/ ALTER / DELETE / UPDATE Sql statements
        - If the query involves multiple SELECT statements with UNION or UNION ALL, make sure each SELECT statement returns the same number of columns.
        - If necessary, add NULL placeholders to ensure consistency in the number of columns for each SELECT.
        - Do not include any explanations or additional text. Only return the final SQL query.
        - dont show ```sql in output
        """

        # Generate the initial SQL query
        response = llm.predict(full_prompt)

        # Clean the response by removing any extra text
        # In this case, you could strip unwanted parts if needed, but the prompt should ideally make this unnecessary
        query = response.strip()

        optimised_query = f"""As an SQL Developer analyse the {query} and return optimized query if necessary based on the query string
                            Intruction:
                            remove "```" from the query string and return the optimized query
                            """

        final_response = llm.predict(full_prompt)

        # Returning the generated SQL query without any extra text
        return final_response

    def execute_query(self, query):
        """Run the SQL query and return results as a table."""
        self.cursor.execute(query)
        columns = [desc[0] for desc in self.cursor.description]
        results = self.cursor.fetchall()
        return pd.DataFrame(results, columns=columns)  # Return DataFrame for tabular format
    
    def visualize_data(self, data):
        """Visualize the data using Sweetviz."""
        # report = sv.analyze(data)
        # report.show_html('sweetviz_report.html')

        # from autoviz import AutoViz_Class
        # AV = AutoViz_Class()
        # return AV.AutoViz(query_result)

        import dtale
        d = dtale.show(data)
        return d.open_browser()

    def close_connection(self):
        """Close the database connection."""
        self.cursor.close()
        self.conn.close()

# Example Usage
if __name__ == "__main__":
    # Initialize DB connection (Use credentials from main.py)
    db = PostgresDBQueryHandler(
                    host="localhost",
                    port=5432,
                    dbname="postgres",
                    user="postgres",
                    password="5010"
                    )
    
    # Fetch Schema Info
    schema_info = db.get_schema_info()

    # print(schema_info)
    
    # Generate SQL Query
    user_prompt = "helloe"
    generated_sql = db.generate_sql_query(
                                        model_choice="gemini-1.5-flash",
                                        prompt=user_prompt,
                                        schema_info=schema_info
                                        )
    
    print(generated_sql)
    
    # Execute SQL Query
    query_result = db.execute_query(generated_sql)
    print(query_result)
    
    # Visualize Data
    db.visualize_data(query_result)
    
    # Close Connection
    db.close_connection()