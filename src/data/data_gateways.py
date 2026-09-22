
import pandas as pd
from typing import Optional, List
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from utilities import utils


class DataGateway:
    def __init__(self, connectionString):
        self.connectionString = connectionString
        self.connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": self.connectionString})
        self.engine = create_engine(self.connection_url, connect_args={"use_setinputsizes": False})

class TblTradingStocksDataGateway(DataGateway):
    def __init__(self, connectionString):
        super().__init__(connectionString)

    def get_stock_data_by_security_symbol(self, security_symbols):
        sql = f"""
            SELECT
                Cusip_Sedol,
                Security_Symbol,
                Date_Entered 
            FROM [DFASys2].[dbo].[tblTrading_Stocks] 
            WHERE Security_Symbol In ({ ", ".join(f"'{s}'" for s in tuple(security_symbols))})
            """

        df = utils.sql_to_df(sql, self.engine)
        df = df.loc[df.groupby('Security_Symbol')['Date_Entered'].idxmax()]
        return df  
    
class TblTradingStocksPricesDataGateway(DataGateway):
    def __init__(self, connectionString):
        super().__init__(connectionString)

    def get_stock_prices_by_cusip_sedols(self, start_date, end_date, cusip_sedols):
        sql = f"""
            SELECT
                Price_Date,
                Cusip_Sedol,
                Price_Adjusted,
                Price_Unadjusted     
            FROM [DFASys2].[dbo].[tblTrading_Stocks_Prices]
            WHERE Cusip_Sedol In ({ ", ".join(f"'{s}'" for s in tuple(cusip_sedols))})
            AND Price_Date Between '{start_date}'
            AND '{end_date}'
            """

        df = utils.sql_to_df(sql, self.engine)
        return df
    
class VwwtblTradingStocksPricesHistoryDataGateway(DataGateway):
    def __init__(self, connectionString):
        super().__init__(connectionString)

    def get_stock_prices_by_cusip_sedols(self, start_date, end_date, cusip_sedols):
        sql = f"""
            SELECT
                Price_Date,
                Cusip_Sedol,
                Price_Adjusted,
                Price_Unadjusted     
            FROM [DFASys2].[dbo].[vwwtblTrading_Stocks_Prices_History]
            WHERE Cusip_Sedol In ({ ", ".join(f"'{s}'" for s in tuple(cusip_sedols))})
            AND Price_Date Between '{start_date}'
            AND '{end_date}'
            """

        df = utils.sql_to_df(sql, self.engine)
        return df        