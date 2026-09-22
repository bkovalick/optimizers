import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import scipy.linalg as la
import yfinance as yf

from .data_gateways import TblTradingStocksDataGateway, TblTradingStocksPricesDataGateway, VwwtblTradingStocksPricesHistoryDataGateway
from utilities.connection_strings import portfolioBreaksCnxn, ds2Cnxn

class Data(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def model_data(self):
        raise NotImplementedError
    
    def _get_stock_returns(self, df_ts: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentage returns from price data."""
        return df_ts.pct_change().iloc[1:] * 100

    def _get_cov_matrix(self, df_r: pd.DataFrame) -> pd.DataFrame:
        """Calculate the covariance matrix of returns."""
        return df_r.cov()
    
    def _get_mean(self, df_r: pd.DataFrame) -> pd.Series:
        """Calculate the mean returns using a shrinkage estimator."""
        if len(df_r) == 0:
            raise ValueError("Input DataFrame is empty. Cannot compute mean returns.")
        
        mu = df_r.mean()
        sigma = self._get_cov_matrix(df_r)
        t, n = df_r.shape
        sigma_shrink = (t - 1) / (t - n - 2) * sigma
        z = la.solve(sigma_shrink, np.ones(n), assume_a="pos")
        mu_0 = mu @ z / z.sum()
        d = mu - mu_0 * np.ones(n)
        y = la.solve(sigma_shrink, d, assume_a="pos")
        w = (n + 2) / (n + 2 + t * d @ y)
        mu_s = (1 - w) * mu + w * mu_0 * np.ones(n)
        return mu_s
    
    def _get_market_caps(self, tickers: list) -> pd.Series:
        market_caps = {}
        for ticker in tickers:
            try:
                stock_info = yf.Ticker(ticker).info
                market_caps[ticker] = stock_info.get("marketCap", np.nan)
            except Exception as e:
                print(f"Error fetching market cap for {ticker}: {e}")
                market_caps[ticker] = np.nan
        return pd.Series(market_caps)

class GurobiModelData(Data):
    def __init__(self, udf_prices = None):
        super().__init__()
        self.udf_prices = udf_prices

    def _get_stock_data(self):
        if self.udf_prices is None:
            return pd.read_pickle("data/subset_weekly_closings_10yrs.pkl")
        else:
            if isinstance(self.udf_prices, str):
                return pd.read_csv(self.udf_prices, index_col=0, parse_dates=True)
            elif isinstance(self.udf_prices, pd.DataFrame):
                return self.udf_prices
            elif isinstance(self.udf_prices, dict):
                return pd.DataFrame.from_dict(self.udf_prices, orient='index')
            elif isinstance(self.udf_prices, list):
                return pd.DataFrame(self.udf_prices)
            else:
                raise ValueError("Unsupported data format for udf_prices. Please provide a DataFrame, dict, or list.")

    @property
    def model_data(self):
        prices = self._get_stock_data()
        returns = self._get_stock_returns(prices)
        covariance_matrix = self._get_cov_matrix(returns)
        mean_returns = self._get_mean(returns)
        return {
            "Prices": prices,
            "Returns": returns,
            "CovMatrix": covariance_matrix,
            "Mean": mean_returns,
            "MarketCaps": None #self._get_market_caps(prices.columns.tolist())
        }

class LiveMarketData(Data):
    def __init__(self):
        super().__init__()
        self.trading_stocks_datagateway = TblTradingStocksDataGateway(ds2Cnxn)
        self.trading_stocks_prices_datagateway = TblTradingStocksPricesDataGateway(ds2Cnxn)
        self.trading_stocks_historical_prices_datagateway = VwwtblTradingStocksPricesHistoryDataGateway(ds2Cnxn)

    @property
    def model_data(self):
        self.trading_stocks = self.trading_stocks_datagateway.get_stock_data_by_security_symbol(self.request["Tickers"])
        self.trading_stocks = self.trading_stocks.drop(columns=["Date_Entered"])
        self.cusip_sedols = self.trading_stocks["Cusip_Sedol"].to_list()
        self.trading_stocks_prices = self.trading_stocks_prices_datagateway\
            .get_stock_prices_by_cusip_sedols(self.request["StartDate"], self.request["EndDate"], self.cusip_sedols)
        self.trading_stocks_historical_prices = self.trading_stocks_historical_prices_datagateway\
            .get_stock_prices_by_cusip_sedols(self.request["StartDate"], self.request["EndDate"], self.cusip_sedols)
        self.trading_stocks_prices = pd.concat([self.trading_stocks_prices, self.trading_stocks_historical_prices]) \
            .drop_duplicates() \
            .reset_index(drop=True)
        self.trading_stocks_prices = pd.merge(self.trading_stocks_prices, self.trading_stocks, on=["Cusip_Sedol"])
        self.model_data["Prices"] = self.trading_stocks_prices.pivot(index=['Price_Date'], columns='Security_Symbol', values='Price_Adjusted')
        self.model_data["Returns"] = self._get_stock_returns(self.model_data["Prices"])
        self.model_data["CovMatrix"] = self._get_cov_matrix(self.model_data["Returns"])
        self.model_data["Mean"] = self._get_mean(self.model_data["Returns"])            
        return self.model_data

if __name__ == '__main__':
    pass