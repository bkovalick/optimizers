"""
===============================================================================
SQL Connection Helper File
===============================================================================
"""

import pandas as pd
import pyodbc as py
from pandas.io.sql import read_sql

import platform
from typing import Optional, List
from urllib.parse import quote_plus


def clean_str(string):
    newstring = []
    for a in string.split(','):
        if a in [' nan', ' NaT', ' None']:
            newstring.append(' NULL')
        elif a in [' nan)', ' None)']:
            newstring.append(' NULL)')
        else:
            newstring.append(a)
    return ','.join(newstring)

def clean_df(df: pd.DataFrame):
    return df

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def cnxnstring(server: str, database: str) -> str:
    """
    :param server: Server for Connection
    :param database: Database for Querying
    :return: ODBC Connection String
    """
    if platform.system != 'Linux':
        driver = "SQL Server"
    else:
        driver = "ODBC Driver 17 for SQL Server"
    conn = (f"Driver={driver};Server={server};Database={database};Trusted_Connection=yes;")
    return conn

def df_to_sql(df: pd.DataFrame, sql: str, engine):
    df = clean_df(df)
    conn = engine.connect().connection
    cursor = conn.cursor()

    records = [str(tuple(x)) for x in df.values]
    for batch in chunker(records, 1):
        newbatch = [clean_str(item) for item in batch]
        rows = ','.join(newbatch)
        insert_rows = sql + rows
        cursor.execute(insert_rows)
        conn.commit()

def sql_to_df(sql_query: str, engine) -> pd.DataFrame:
    """ Executes a single query and returns results in a dataframe """
    with engine.begin() as cnxn:
        df = read_sql(sql_query, cnxn)
    
    for col in df.columns:
        if df[col].dtype=='object':
            df.loc[:,col] = df[col].str.strip()
        else:
            pass 

    return df

def mult_queries_to_dfs(constr: str, sql_query_list: List[str]) -> List[pd.DataFrame]:
    """ Execute multiple queries within the same connection and returns results as a list of dataframes """
    dfList = []
    with py.connect(constr) as cnxn:
        for sql in sql_query_list:
            dfList.append(read_sql(sql, cnxn))
    return dfList

if __name__ == "__main__":
    print(cnxnstring("PortfolioBreaksDB-PRD", "PortfolioBreaks"))

