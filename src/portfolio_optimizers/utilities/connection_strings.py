from portfolio_optimizers.utilities import utils

portCharsDataCnxn = utils.cnxnstring("PortfolioCharsDB-PRD", "PortfolioChars")
ds2Cnxn = utils.cnxnstring("DFASys2DB-PRD", "DFASys2")
ds2ArchiveCnxn = utils.cnxnstring("DFASys2DB-PRD", "DFASys2_Archive")
buyRangeDataCnxn = utils.cnxnstring("BuyRangeDataDB-PRD", "BuyRangeData")
portfolioBreaksCnxn = utils.cnxnstring("PortfolioBreaksDB-PRD", "PortfolioBreaks")