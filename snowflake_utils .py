import os
import pandas as pd
from snowflake.connector import connect
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
from data_science_utils import getcode, get_connection, upload_large_table

def get_snowflake_connection(database, schema, role, warehouse):
    """
    Establish a connection to Snowflake.

    Parameters:
    database (str): The name of the database.
    schema (str): The schema to be used.
    role (str): The role to be used.
    warehouse (str): The warehouse to be used.

    Returns:
    Connection object: A connection object to interact with Snowflake.
    """
    print("Establishing Snowflake connection...")
    connection = get_connection(database=database, schema=schema, role=role, warehouse=warehouse)
    print("Connection established.")
    return connection

def load_data(connection, query):
    """
    Load data from Snowflake using a specified query.

    Parameters:
    connection (Connection object): The Snowflake connection object.
    query (str): The SQL query to execute.

    Returns:
    DataFrame: The loaded data as a pandas DataFrame.
    """
    print(f"Loading data with query: {query}")
    data = getcode(query, connection=connection)
    print(f"Data loaded. Number of rows: {len(data)}")
    return data

def upload_data(connection, dataframe, table_name, if_exists='replace'):
    """
    Upload data to a Snowflake table.

    Parameters:
    connection (Connection object): The Snowflake connection object.
    dataframe (DataFrame): The pandas DataFrame to upload.
    table_name (str): The name of the target table in Snowflake.
    if_exists (str): Behavior when the table already exists. Default is 'replace'.
    """
    print(f"Uploading data to table: {table_name}")
    upload_large_table(connection, dataframe, table_name, if_exists=if_exists)
    print("Data uploaded successfully.")
