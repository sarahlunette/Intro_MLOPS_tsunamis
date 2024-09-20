''' TODO: Document functions'''
import csv
import psycopg2

# Database connection parameters
''' TODO: Make env variables with the database credentials'''
db_config = {
    "host": "your_host",
    "database": "your_dbname",
    "user": "your_username",
    "password": "your_password",
    "port": "your_port"
}

# Function to create a connection to the PostgreSQL database
def create_connection(db_config):
    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
            port=db_config["port"]
        )
        return conn
    except Exception as error:
        print(f"Error connecting to PostgreSQL: {error}")
        return None

# Function to create table (adjust columns as per your CSV)

''' TODO: Change column names to tsunamis names, the acquisition of data can select columns before writing to database'''
def create_table(conn):
    query = """
    CREATE TABLE IF NOT EXISTS tsunamis (
        id SERIAL PRIMARY KEY,
        column1 TEXT,
        column2 TEXT,
        column3 TEXT,
        column4 INTEGER
    );
    """
    with conn.cursor() as cur:
        cur.execute(query)
        conn.commit()

# Function to insert data from CSV into the PostgreSQL table
def insert_data_from_csv(conn, csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row if present
        with conn.cursor() as cur:
            for row in reader:
                cur.execute(
                    "INSERT INTO your_table_name (column1, column2, column3, column4) VALUES (%s, %s, %s, %s);",
                    row
                )
  
