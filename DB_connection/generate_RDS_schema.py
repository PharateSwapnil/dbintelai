import subprocess
 
def dump_schema_to_variable(host, user, database):
    command = f"pg_dump -h {host} -U {user} -d {database} -s"
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        schema_output = result.stdout
        return schema_output
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None
 
# Usage
schema_sql = dump_schema_to_variable("localhost", "myuser", "mydatabase")
 
if schema_sql:
    print(schema_sql[:500])  # Print first 500 characters for preview
 
 
# save in txt
 
def dump_schema_to_file(host, user, database, output_file):
    schema_sql = dump_schema_to_variable(host, user, database)
    if schema_sql:
        with open(output_file, "w") as f:
            f.write(schema_sql)
        print(f"Schema dump saved to {output_file}")
 
# Usage
dump_schema_to_file("localhost", "myuser", "mydatabase", "schema_dump.txt")