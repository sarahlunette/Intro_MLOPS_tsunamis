from create_db import insert_data_from_csv, create_connection

csv_file = path

''' TODO: replace with the right credentials for the interim db'''
db_config = {
    "host": "your_host",
    "database": "your_dbname",
    "user": "your_username",
    "password": "your_password",
    "port": "your_port"
}

conn = create_connection(db_config)
insert_data_from_csv(conn, csv_file)
