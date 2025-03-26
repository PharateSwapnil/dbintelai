import os
import logging
import json
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# PostgreSQL Connector
class PostgreSQLConnector:
    def __init__(self, config=None):
        self.connection = None
        self.config = config or {}
        
        # If no config is provided, use environment variables
        if not config:
            self.config = {
                'host': os.environ.get('PGHOST', 'localhost'),
                'port': os.environ.get('PGPORT', 5432),
                'database': os.environ.get('PGDATABASE', 'postgres'),
                'user': os.environ.get('PGUSER', 'postgres'),
                'password': os.environ.get('PGPASSWORD', '')
            }
    
    def connect(self):
        """Establish a connection to the PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password']
            )
            logger.info(f"Successfully connected to PostgreSQL at {self.config['host']}:{self.config['port']}/{self.config['database']}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {str(e)}")
            return False
    
    def execute_query(self, query):
        """Execute a SQL query and return the results."""
        if not self.connection:
            success = self.connect()
            if not success:
                return {
                    "error": "Could not connect to PostgreSQL database",
                    "results": []
                }
        
        cursor = None
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query)
            
            # Check if it's a SELECT query (has results to fetch)
            if query.strip().lower().startswith('select'):
                results = cursor.fetchall()
                # Convert results to JSON-compatible format
                json_results = [dict(row) for row in results]
                return {
                    "success": True,
                    "results": json_results
                }
            else:
                # For non-SELECT queries, commit the changes
                if self.connection:
                    self.connection.commit()
                return {
                    "success": True,
                    "affected_rows": cursor.rowcount
                }
        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return {
                "error": str(e),
                "results": []
            }
        finally:
            if cursor:
                cursor.close()
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("PostgreSQL connection closed")
    
    def get_schema(self):
        """Get the database schema."""
        schema_query = """
        SELECT 
            t.table_name, 
            c.column_name, 
            c.data_type,
            c.is_nullable
        FROM 
            information_schema.tables t
        JOIN 
            information_schema.columns c 
        ON 
            t.table_name = c.table_name
        WHERE 
            t.table_schema = 'public'
        ORDER BY 
            t.table_name, 
            c.ordinal_position;
        """
        
        result = self.execute_query(schema_query)
        if "error" in result:
            return result
        
        # Organize the schema information by table
        schema = {}
        for row in result["results"]:
            table_name = row["table_name"]
            if table_name not in schema:
                schema[table_name] = []
            
            schema[table_name].append({
                "column_name": row["column_name"],
                "data_type": row["data_type"],
                "is_nullable": row["is_nullable"]
            })
        
        return {
            "success": True,
            "schema": schema
        }

# GraphDB Connector (simulated)
class GraphDBConnector:
    def __init__(self, config=None):
        self.config = config or {}
        self.is_connected = False
        
        # Default configuration if not provided
        if not config:
            self.config = {
                'endpoint': 'http://LP148:7200/repositories/genai_graph_DB'
            }
        # http://LP148:7200/repositories/genai_graph_DB  actual endpoint
        # 'http://localhost:7200/repositories/power_system'
    
    def connect(self):
        """Simulate establishing a connection to GraphDB."""
        try:
            # In a real implementation, we would establish a connection to GraphDB
            # For now, just simulate success
            self.is_connected = True
            logger.info(f"Successfully connected to GraphDB at {self.config['endpoint']}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to GraphDB: {str(e)}")
            return False
    
    def execute_query(self, query):
        """Simulate executing a SPARQL query."""
        # In a real implementation, we would send the query to GraphDB
        # For now, just simulate success with mock data
        try:
            if not self.is_connected:
                success = self.connect()
                if not success:
                    return {
                        "error": "Could not connect to GraphDB",
                        "results": []
                    }
            
            logger.info(f"Simulated executing SPARQL query on GraphDB")
            return {
                "success": True,
                "results": [{"info": "This is a simulated response from GraphDB"}]
            }
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {str(e)}")
            return {
                "error": str(e),
                "results": []
            }
    
    def close(self):
        """Close the connection."""
        self.is_connected = False
        logger.info("GraphDB connection closed")

# Factory function to create the appropriate connector
def get_connector(db_type, config=None):
    """Get a database connector based on the database type."""
    if db_type == "postgres" or db_type == "aws-rds":
        return PostgreSQLConnector(config)
    elif db_type == "graphdb" or db_type == "aws-neptune":
        return GraphDBConnector(config)
    else:
        raise ValueError(f"Unsupported database type: {db_type}")