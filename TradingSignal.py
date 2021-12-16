import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from EFPortfolio import EFPortfolio
from LassoPortfolio import LassoPortfolio
from Utility import print_full


def backtest(fileSuffix, basePortfolio, initial_balance, price_df, trade_date, type="BASE"):
    columns = np.append(price_df.columns.values, np.array(['cash', 'balance']))
    trading_df = pd.DataFrame(index=trade_date, columns=columns)

    initial_date = trade_date[0]
    # initial_date = datetime.strptime(trade_date[0], "%Y-%m-%d")

    applyPortfolio = None

    # start_day = (initial_date - relativedelta(years=1)).date()
    start_day = (initial_date - relativedelta(months=3)).date()
    end_day = (initial_date - relativedelta(days=1)).date()

    if type == "BASE":
        applyPortfolio = basePortfolio

    elif type == "LASSO":
        # lassoPortfolio = {}
        # for asset in basePortfolio.columns:
        #     lassoPortfolio[asset] = basePortfolio.loc[initial_date][asset]
        portfolio = LassoPortfolio(basePortfolio.keys(), start_day, end_day)
        lassoPortfolio = portfolio.run()

        applyPortfolio = lassoPortfolio

    elif type == "EF_MAX_SHARPE":
        portfolio = EFPortfolio(basePortfolio.keys(), start_day, end_day)
        efPortfolio = portfolio.run('MAX_SHARPE')
        applyPortfolio = efPortfolio

    else:
        print("unknown type ...")

    # print(f"... {start_day} ~ {end_day} ...", applyPortfolio)

    sumAmount = 0
    for member in applyPortfolio.keys():
        targetAmount = initial_balance * applyPortfolio[member]
        buyAmount = math.floor(targetAmount / price_df.loc[initial_date][member])

        sumAmount = sumAmount + (buyAmount * price_df.loc[initial_date][member])
        trading_df.loc[initial_date][member] = buyAmount

    extraAmount = initial_balance - sumAmount
    trading_df.loc[initial_date]['cash'] = extraAmount
    trading_df.loc[initial_date]['balance'] = initial_balance

    for i in range(1, len(trade_date)):
        befDate = trade_date[i - 1]
        curDate = trade_date[i]

        running_balance = trading_df.loc[befDate]['cash']
        for member in applyPortfolio.keys():
            running_balance += trading_df.loc[befDate][member] * price_df.loc[curDate][member]

        ## UPDATE Portfolio
        applyPortfolio = None

        # start_day = (curDate - relativedelta(years=1)).date()
        start_day = (curDate - relativedelta(months=3)).date()
        end_day = (curDate - relativedelta(days=1)).date()

        if type == "BASE":
            applyPortfolio = basePortfolio

        elif type == "LASSO":
            # lassoPortfolio = {}
            # for asset in basePortfolio.columns:
            #     lassoPortfolio[asset] = basePortfolio.loc[curDate][asset]

            portfolio = LassoPortfolio(basePortfolio.keys(), start_day, end_day)
            lassoPortfolio = portfolio.run()

            applyPortfolio = lassoPortfolio

        elif type == "EF_MAX_SHARPE":
            portfolio = EFPortfolio(basePortfolio.keys(), start_day, end_day)
            efPortfolio = portfolio.run('MAX_SHARPE')
            applyPortfolio = efPortfolio

        else:
            print("unknown type ...")

        # print(f"... {start_day} ~ {end_day} ...", applyPortfolio)

        sumAmount = 0
        for member in applyPortfolio.keys():
            # print( f'{member} : {price_df.loc[curDate][member]}')

            targetAmount = running_balance * applyPortfolio[member]
            buyAmount = math.floor(targetAmount / price_df.loc[curDate][member])

            sumAmount = sumAmount + (buyAmount * price_df.loc[curDate][member])
            trading_df.loc[curDate][member] = buyAmount

        extraAmount = running_balance - sumAmount
        trading_df.loc[curDate]['cash'] = extraAmount
        trading_df.loc[curDate]['balance'] = running_balance

        # print( f'... CASH {extraAmount} : {running_balance} - {sumAmount} ... ' )

    return trading_df


def calcFullData(tradingDF, basePortfolio, price_df):
    daily_df = pd.DataFrame(index=price_df.index, columns=tradingDF.columns)

    for dateIdx in daily_df.index.tolist():
        baseDate = tradingDF.loc[:dateIdx].index[-1]

        for member in basePortfolio.keys():
            daily_df.loc[dateIdx][member] = price_df.loc[dateIdx][member] * tradingDF.loc[baseDate][member]

        daily_df.loc[dateIdx]['cash'] = tradingDF.loc[baseDate]['cash']
        daily_df.loc[dateIdx]['balance'] = np.sum(daily_df.loc[dateIdx])

    daily_df = daily_df.dropna()

    daily_df['Daily-Ret'] = daily_df['balance'].pct_change()
    daily_df['Cum-Ret'] = (1 + daily_df['Daily-Ret']).cumprod() - 1

    return daily_df[['balance', 'Daily-Ret', 'Cum-Ret']]


def visualizePerf(baseFullDF, efFullDF, lassoFullDF):
    plt.figure(figsize=(16, 9))
    ax = plt.subplot()

    plt.plot(baseFullDF['Cum-Ret'].index, baseFullDF['Cum-Ret'], color='lightgray', label='Fixed')
    plt.plot(efFullDF['Cum-Ret'].index, efFullDF['Cum-Ret'], color='blue', label='MPT(Max Sharpe)')
    plt.plot(lassoFullDF['Cum-Ret'].index, lassoFullDF['Cum-Ret'], color='red', label='Lasso')

    plt.legend(loc='best')
    plt.ylabel('Price')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))

    plt.show()


def checkTradingSignal(weightDF, priceDF, dirPath):
    print('투자 의견 ... ')
    # 기준 평가액
    initial_balance = 10_000_000

    basePortfolio = {}
    cntAsset = len(weightDF.columns)
    fixedWeight = 1 / cntAsset
    for asset in weightDF.columns:
        basePortfolio[asset] = fixedWeight

    trade_date = weightDF.index.tolist()

    # 위의 리스트에서 6개월 이상 지난 것은 제외 : 6 개월 단위로 점검되도록 수정 ...
    baseTimestamp = trade_date[0] - pd.offsets.DateOffset(months=6)
    trade_date = [x for x in trade_date if x > baseTimestamp]

    baseBalanceDF = backtest(dirPath, basePortfolio, initial_balance, priceDF, trade_date, "BASE")
    baseFullDF = calcFullData(baseBalanceDF, basePortfolio, priceDF)

    efBalanceDF = backtest(dirPath, basePortfolio, initial_balance, priceDF, trade_date, "EF_MAX_SHARPE")
    efFullDF = calcFullData(efBalanceDF, basePortfolio, priceDF)

    lassoBalanceDF = backtest(dirPath, basePortfolio, initial_balance, priceDF, trade_date, "LASSO")
    lassoFullDF = calcFullData(lassoBalanceDF, basePortfolio, priceDF)

    # print_full( lassoFullDF )

    visualizePerf(baseFullDF, efFullDF, lassoFullDF)

    # ['balance', 'Daily-Ret', 'Cum-Ret']
    benchBase = baseFullDF.iloc[-1:]['Cum-Ret'].values[0]
    benchEF = efFullDF.iloc[-1:]['Cum-Ret'].values[0]
    lassoRet = lassoFullDF.iloc[-1:]['Cum-Ret'].values[0]

    print( benchBase, benchEF, lassoRet )

    if lassoRet < benchEF and lassoRet < benchBase:
        minRet = min( benchEF, benchBase)
        diffRet = abs( lassoRet - minRet )
        if diffRet > 0.05:
            print( f'매도')

    if lassoRet > benchEF and lassoRet > benchBase:
        print( '매수 추천 ...')


if __name__ == '__main__':
    dirPath = f"./2021-11-18"

    investDF = pd.read_csv(f"{dirPath}/INVEST.csv", index_col=0)
    investDF.index = pd.to_datetime(investDF.index)

    weightDF = pd.read_csv(f"{dirPath}/PORTFOLIO.csv", index_col=0)
    weightDF.index = pd.to_datetime(weightDF.index)

    priceDF = pd.read_csv(f"{dirPath}/PRC_CLOSE.csv", index_col=0)
    priceDF.index = pd.to_datetime(priceDF.index)

    checkTradingSignal(weightDF, priceDF, dirPath)
