import math
from datetime import date

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import pandas_datareader as pdr
from dateutil.relativedelta import relativedelta

from LassoPortfolio import LassoPortfolio
from EFPortfolio import EFPortfolio
from Utility import print_full


def loadAsset(assetName, filePrefix, start_day, end_day):
    # print(f"./data/{filePrefix}_{assetName}.csv")

    df = pd.read_csv(f"./{filePrefix}/PRC_{assetName}.KS.csv", index_col=0)
    df.index = pd.to_datetime(df.index)

    df = df.rename({'close': assetName}, axis='columns')
    filterDF = df.loc[start_day:end_day]

    return filterDF


def make_trade_date(kospi_index, rebalancing_date):
    # 날짜 리스트 생성
    date = []
    date_index = kospi_index.index

    for idx in range(len(date_index)):
        date.append(date_index[idx])

    # 거래일 데이터 생성
    trade_date = [date[0]]
    for i in range(1, len(date)):
        if rebalancing_date == 1:  # 첫 거래일에 리밸런싱 한다면,
            if date[i] == date[-1] or date[i].month != date[i - 1].month:  # 오늘 월과 어제 월이 다르거나, 마지막 거래일이면,
                trade_date.append(date[i])
        elif rebalancing_date == 0:  # 마지막 거래일에 리밸런싱 한다면,
            if date[i] == date[-1] or date[i].month != date[i + 1].month:  # 오늘 월과 내일 월이 다르거나, 마지막 거래일이면,
                trade_date.append(date[i])
        elif rebalancing_date <= date[i].day:  # 특정일에 리밸런싱 한다면, 오늘 날짜가 특정일보다 크거나 같으면,
            if date[i] == date[-1] or trade_date[len(trade_date) - 1].month != date[i - 1].month:  # 전 거래월과 오늘월이 다르다면,
                trade_date.append(date[i])

    time_period = (trade_date[-1] - trade_date[0]).days / 365  # 거래 기간 계산

    return trade_date, time_period


def backtest(fileSuffix, basePortfolio, initial_balance, price_df, trade_date, type="BASE"):
    columns = np.append(price_df.columns.values, np.array(['cash', 'balance']))
    trading_df = pd.DataFrame(index=trade_date, columns=columns)

    initial_date = trade_date[0]

    applyPortfolio = None

    if type == "BASE":
        applyPortfolio = basePortfolio
        print(applyPortfolio)
    else:
        # start_day = (initial_date - relativedelta(years=1)).date()
        # start_day = (initial_date - relativedelta(months=6)).date()
        # start_day = (initial_date - relativedelta(months=3)).date()
        start_day = (initial_date - relativedelta(months=1)).date()
        end_day = (initial_date - relativedelta(days=1)).date()

        if type == "EF_MAX_SHARPE":
            portfolio = EFPortfolio(basePortfolio.keys(), start_day, end_day, fileSuffix)
            efPortfolio = portfolio.run('MAX_SHARPE')
            applyPortfolio = efPortfolio

        elif type == "LASSO":
            portfolio = LassoPortfolio(basePortfolio.keys(), start_day, end_day, fileSuffix)
            lassoPortfolio = portfolio.run()
            applyPortfolio = lassoPortfolio

        print(f"... {start_day} ~ {end_day} ...", applyPortfolio)

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

        if type == "BASE":
            applyPortfolio = basePortfolio
        else:
            # start_day = (curDate - relativedelta(years=1)).date()
            # start_day = (curDate - relativedelta(months=6)).date()
            # start_day = (curDate - relativedelta(months=3)).date()
            start_day = (curDate - relativedelta(months=1)).date()
            end_day = (curDate - relativedelta(days=1)).date()

            if type == "EF_MAX_SHARPE":
                portfolio = EFPortfolio(basePortfolio.keys(), start_day, end_day, fileSuffix)
                efPortfolio = portfolio.run('MAX_SHARPE')
                applyPortfolio = efPortfolio

            elif type == "LASSO":
                portfolio = LassoPortfolio(basePortfolio.keys(), start_day, end_day, fileSuffix)
                lassoPortfolio = portfolio.run()
                applyPortfolio = lassoPortfolio

            print(f"... {start_day} ~ {end_day} ...", applyPortfolio)

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


def calcMetric(inFullDF):
    ## Metrics
    metrics = [
        'Annual Return',  # return on the investment in total
        'Cumulative Returns',  # return on investment received that year
        'Annual Volatility',  # daily volatility times the square root of 252 trading days
        'Sharpe Ratio',
        # measures the performance of an investment compared to a risk-free asset, after adjusting for its risk
        'Sortino Ratio'
        # differentiates harmful volatility from total overall volatility by using the asset's standard deviation of ...
    ]

    columns = ['Result']
    portEvalDF = pd.DataFrame(index=metrics, columns=columns)

    portEvalDF.loc['Cumulative Returns'] = inFullDF['Cum-Ret'][-1]
    portEvalDF.loc['Annual Return'] = inFullDF['Daily-Ret'].mean() * 252
    portEvalDF.loc['Annual Volatility'] = inFullDF['Daily-Ret'].std() * np.sqrt(252)
    portEvalDF.loc['Sharpe Ratio'] = (inFullDF['Daily-Ret'].mean() * 252) / (inFullDF['Daily-Ret'].std() * np.sqrt(252))

    ## Calculate Downside Return
    sortino_ratio_df = inFullDF[['Daily-Ret']].copy()
    sortino_ratio_df.loc[:, 'Downside Returns'] = 0

    target = 0
    mask = sortino_ratio_df['Daily-Ret'] < target
    sortino_ratio_df.loc[mask, 'Downside Returns'] = sortino_ratio_df['Daily-Ret'] ** 2

    ## Calculate Sortino Ratio
    down_stdev = np.sqrt(sortino_ratio_df['Downside Returns'].mean()) * np.sqrt(252)
    expected_return = sortino_ratio_df['Daily-Ret'].mean() * 252
    sortino_ratio = expected_return / down_stdev

    portEvalDF.loc['Sortino Ratio'] = sortino_ratio

    return portEvalDF


def mergePerf(basePerfDF, efPerDF, lassoPerfDF):
    columns = ['Fixed', 'MPT', 'Lasso']
    mergePerfDF = pd.DataFrame(index=basePerfDF.index, columns=columns)

    mergePerfDF['Fixed'] = basePerfDF
    mergePerfDF['MPT'] = efPerDF
    mergePerfDF['Lasso'] = lassoPerfDF

    return mergePerfDF


def visualizePerf(baseFullDF, efFullDF, lassoFullDF):

    plt.figure(figsize=(16, 9))
    ax = plt.subplot()

    plt.plot(baseFullDF['Cum-Ret'].index, baseFullDF['Cum-Ret'], color='lightgray', label='Fixed')
    plt.plot(efFullDF['Cum-Ret'].index, efFullDF['Cum-Ret'], color='blue', label='MPT(Max Sharpe)')
    plt.plot(lassoFullDF['Cum-Ret'].index, lassoFullDF['Cum-Ret'], color='red', label='Lasso')

    plt.legend(loc='best')
    plt.ylabel('Price')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))

    plt.show()


if __name__ == '__main__':

    # 백테스트 날짜 설정
    today = date(2019, 11, 18)
    start_day = today - relativedelta(years=1)
    end_day = today - relativedelta(days=1)

    # 최초 평가액
    initial_balance = 10_000_000

    # 리밸런싱 날짜
    rebalancing_date = 0  # 0은 마지막 거래일, 1은 첫 거래일, 기타 숫자는 특정일

    # 기준 날짜를 위한 코스피 지수 ...
    index_df = pdr.get_data_yahoo("^KS11", start_day, end_day)
    trade_date, time_period = make_trade_date(index_df, rebalancing_date)

    print(f"백테스트 날짜 : {start_day} ~ {end_day}")

    # 포트폴리오 구성종목
    # basePortfolio = {'114100.KS': 0.3,  # 국내채권, KBSTAR 국고채3년
    #                  '132030.KS': 0.2,  # 금,     KODEX 골드선물(H)
    #                  '261240.KS': 0.3,  # 달러,    KODEX 미국달러선물
    #                  '161510.KS': 0.1,  # 배당,    ARIRANG 고배당주
    #                  '069500.KS': 0.1}  # 국내주식, KODEX 200

    basePortfolio = {
        '152100': 0.04,
        '102110': 0.04,
        # '329650': 0.03,
        # '329750': 0.03,
        '153130': 0.04,
        # '360200': 0.03,
        '219480': 0.04,
        '273130': 0.04,
        # '360750': 0.03,
        '292150': 0.04,
        # '314250': 0.03,
        '105190': 0.04,
        '214980': 0.04,
        # '367380': 0.03,
        '091170': 0.04,
        '278540': 0.04,
        '219390': 0.04,
        '102780': 0.04,
        '261220': 0.04,
        '132030': 0.04,
        '192090': 0.04,
        '069500': 0.04,
        '148020': 0.04,
        '278530': 0.04,
        '157450': 0.04,
        '275980': 0.04,
        '133690': 0.04,
        '144600': 0.04,
        '251350': 0.04,
        '196230': 0.04
    }

    # 시세 데이터 로딩
    price_df = pd.DataFrame()
    price_df['date'] = index_df.index
    price_df.set_index('date', inplace=True)

    fileSuffix = 'Profile/JYLEE'
    for member in basePortfolio.keys():
        price_df = price_df.join(loadAsset(member, fileSuffix, start_day, end_day)[member])

    # price_df = price_df.fillna(method='ffill')  # 중간 NaN 이전 값으로 채우기
    # price_df = price_df.dropna()  # NaN 있는 행 제거 (최초 거래일 이전 날짜 모두 제거)
    # print('price_df : 가격 데이터프레임')
    # print_full( price_df )
    price_df.to_csv(f"{fileSuffix}/PRC_CLOSE.csv")

    print('\n* 고정 비율 분배')
    print(100 * '-')
    baseBalanceDF = backtest(fileSuffix, basePortfolio, initial_balance, price_df, trade_date, "BASE")
    print()
    print_full(baseBalanceDF)
    baseBalanceDF.to_csv(f"{fileSuffix}/RS_FIXED.csv")

    baseFullDF = calcFullData(baseBalanceDF, basePortfolio, price_df)
    basePerfDF = calcMetric(baseFullDF)
    # print()
    # print_full( baseFullDF )
    baseFullDF.to_csv(f"{fileSuffix}/RS_FULL_FIXED.csv")

    print('\n* Efficient Frontier : Max Sharpe')
    print(100 * '-')
    efBalanceDF = backtest(fileSuffix, basePortfolio, initial_balance, price_df, trade_date, "EF_MAX_SHARPE")
    print()
    print_full(efBalanceDF)
    efBalanceDF.to_csv(f"{fileSuffix}/RS_EFSHARPE.csv")

    efFullDF = calcFullData(efBalanceDF, basePortfolio, price_df)
    efPerfDF = calcMetric(efFullDF)
    # print()
    # print_full( efPerfDF )
    efFullDF.to_csv(f"{fileSuffix}/RS_FULL_EFSHARPE.csv")

    print('\n* Lasso ')
    print(100 * '-')
    lassoBalanceDF = backtest(fileSuffix, basePortfolio, initial_balance, price_df, trade_date, "LASSO")
    print()
    print_full(lassoBalanceDF)
    lassoBalanceDF.to_csv(f"{fileSuffix}/RS_LASSO.csv")

    lassoFullDF = calcFullData(lassoBalanceDF, basePortfolio, price_df)
    lassoPerfDF = calcMetric(lassoFullDF)
    # print()
    # print_full( lassoPerfDF )
    lassoFullDF.to_csv(f"{fileSuffix}/RS_FULL_LASSO.csv")

    resultDF = mergePerf(basePerfDF, efPerfDF, lassoPerfDF)
    print()
    print_full(resultDF)
    resultDF.to_csv(f"{fileSuffix}/RS_MERGED.csv")

    visualizePerf(baseFullDF, efFullDF, lassoFullDF)
