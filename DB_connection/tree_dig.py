# pip install networkx matplotlib
 
 
 
import psycopg2
import networkx as nx
import matplotlib.pyplot as plt
 
def get_foreign_keys(host, database, user, password):
    """Fetch foreign key relationships from PostgreSQL."""
    conn = psycopg2.connect(host=host, dbname=database, user=user, password=password)
    cur = conn.cursor()
 
    query = """
    SELECT
        conrelid::regclass AS table_name,
        confrelid::regclass AS referenced_table,
        conname AS constraint_name
    FROM pg_constraint
    WHERE contype = 'f';
    """
    
    cur.execute(query)
    foreign_keys = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return foreign_keys
 
def create_relationship_graph(foreign_keys):
    """Create a directed graph from table relationships."""
    G = nx.DiGraph()
 
    for table, referenced_table, constraint in foreign_keys:
        G.add_edge(referenced_table, table, label=constraint)  # Reference table → Foreign key table
 
    return G
 
def plot_graph(G):
    """Plot the database table relationship graph."""
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)  # Position nodes
 
    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, edge_color='gray', font_size=10, font_weight='bold', arrows=True)
    
    # Draw edge labels
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
 
    plt.title("Database Table Relationship Graph")
plt.show()
 
# Usage
host = "localhost"
database = "mydatabase"
user = "myuser"
password = "mypassword"
 
foreign_keys = get_foreign_keys(host, database, user, password)
G = create_relationship_graph(foreign_keys)
 
if G.nodes:
    plot_graph(G)
else:
    print("No foreign key relationships found.")