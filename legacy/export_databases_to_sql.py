import sqlite3
import os

def export_to_sql(sqlite_path, sql_output_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(sqlite_path)
    
    # Write the database dump to the .sql file
    with open(sql_output_path, 'w') as f:
        for line in conn.iterdump():
            f.write('%s' % line)
    
    conn.close()
    print(f"Exported {sqlite_path} to {sql_output_path}")

# Example usage for one database
db_path = "/home/mindmap/Desktop/SLM/data/raw/databases/academic/academic.sqlite"
output_path = "/home/mindmap/Desktop/SLM/data/raw/databases/academic/academic_export.sql"

if os.path.exists(db_path):
    export_to_sql(db_path, output_path)
else:
    print(f"File not found: {db_path}")
