import math
import os
import re
from datetime import date

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pandas_datareader as pdr

from dateutil.relativedelta import relativedelta

import TradingSignal
from LassoPortfolio import LassoPortfolio
from PreProcess import ClassifyETF, TargetETF
from Utility import print_full


def createFolder(directory):
    current_path = os.getcwd()
    mkdir_path = f"{current_path}/{directory}"

    try:
        if not os.path.exists(mkdir_path):
            os.makedirs(mkdir_path)
    except OSError:
        print('Error: Creating directory. ' + mkdir_path)

    return f"./{directory}"


def getTargetETFs(dirPath):
    portfolio = pd.read_csv(f"{dirPath}/PORTFOLIO.csv", index_col=0)

    return portfolio.columns.values.tolist()
    # path = f'{dirPath}/'
    # files = os.listdir(path)
    #
    # ## 현재 디렉토리에 존재하는 데이터 목록 생성
    # lstMember = []
    # for f in files:
    #     member = re.findall(f'PRC_(.+).KS.csv', f)
    #     if len(member) != 0:
    #         lstMember.extend(member)
    #
    # return lstMember


def getPriceData(lstAsset, idxInitDay, idxToday, dirPath, isRebalance):
    priceDF = pd.DataFrame()
    for member in lstAsset:
        df = pd.read_csv(f"./Assets/PRC_{member}.KS.csv", index_col=0)
        df.index = pd.to_datetime(df.index)

        if isRebalance:
            filterDF = df.loc[idxInitDay:idxToday]
        else:
            filterDF = df.iloc[-1:]

        priceDF[member] = filterDF['close']

    return priceDF


def invest(idxDate, balance, rsDict, priceDF, dirPath, isRebalance, isCheck):
    preInvestDF = None
    curInvestDF = None

    preInvestWeightDF = None
    curInvestWeightDF = None

    running_balance = balance

    if isRebalance:
        preInvestWeightDF = pd.read_csv(f"{dirPath}/PORTFOLIO.csv", index_col=0)
        preInvestWeightDF.index = pd.to_datetime(preInvestWeightDF.index)

        preInvestDF = pd.read_csv(f"{dirPath}/INVEST.csv", index_col=0)
        preInvestDF.index = pd.to_datetime(preInvestDF.index)

        prevRaw = preInvestDF.iloc[-1:]
        running_balance += prevRaw['cash'].values[0]

        for member in rsDict.keys():
            price = priceDF.iloc[-1:][member].values[0]
            running_balance += prevRaw[member].values[0] * price

    columns = np.append(priceDF.columns.values, np.array(['cash', 'balance']))
    curInvestDF = pd.DataFrame(index=[idxDate], columns=columns)

    curInvestWeightDF = pd.DataFrame(index=[idxDate], columns=priceDF.columns)

    sumAmount = 0

    for member in rsDict.keys():
        price = priceDF.iloc[-1:][member].values[0]

        targetAmount = running_balance * rsDict[member]
        buyAmount = math.floor(targetAmount / price)

        curInvestWeightDF.loc[idxDate][member] = rsDict[member]

        if isRebalance and isCheck:
            prevRaw = preInvestDF.iloc[-1:]
            prevWeight = preInvestWeightDF.iloc[-1:]

            buyAmount = prevRaw[member].values[0]
            curInvestWeightDF.loc[idxDate][member] = prevWeight[member].values[0]

        sumAmount = sumAmount + (buyAmount * price)
        curInvestDF.loc[idxDate][member] = buyAmount

    extraAmount = running_balance - sumAmount
    curInvestDF.loc[idxDate]['cash'] = extraAmount
    curInvestDF.loc[idxDate]['balance'] = running_balance

    if isRebalance:
        curInvestDF = curInvestDF.combine_first(preInvestDF)
        curInvestWeightDF = curInvestWeightDF.combine_first(preInvestWeightDF)

    if not isCheck:
        curInvestDF = curInvestDF[columns].astype(int)
        curInvestDF.to_csv(f"{dirPath}/INVEST.csv")

        curInvestWeightDF.to_csv(f"{dirPath}/PORTFOLIO.csv")

    return curInvestDF, curInvestWeightDF, running_balance


def calcFullData(tradingDF, price_df, balance, isRebalance, isCheck, dirPath):
    account_df = None

    if isRebalance:
        account_df = pd.read_csv(f"{dirPath}/ACCOUNT.csv", index_col=0)
        account_df.index = pd.to_datetime(account_df.index)

        account_df.loc[tradingDF.index[-1]] = [ balance, account_df['credit'].sum() + balance ]
        # account_df.loc[tradingDF.index[-1]]['cum_credit'] = account_df['credit'].sum()

        if not isCheck:
            account_df.to_csv(f"{dirPath}/ACCOUNT.csv")
    else:
        account_df = pd.DataFrame(index=tradingDF.index, columns=['credit', 'cum_credit', 'diff', 'return'])

        account_df.loc[tradingDF.index[0]]['credit'] = balance
        account_df.loc[tradingDF.index[0]]['cum_credit'] = balance

        account_df.to_csv(f"{dirPath}/ACCOUNT.csv")

    daily_columns = np.append(tradingDF.columns.values, np.array(['total', 'credit']))
    daily_df = pd.DataFrame(index=price_df.index, columns=daily_columns)

    for dateIdx in daily_df.index.tolist():
        baseDate = tradingDF.loc[:dateIdx].index[-1]

        for member in tradingDF.columns:
            if member == 'cash' or member == 'balance':
                continue

            daily_df.loc[dateIdx][member] = price_df.loc[dateIdx][member] * tradingDF.loc[baseDate][member]

        daily_df.loc[dateIdx]['cash'] = tradingDF.loc[baseDate]['cash']
        daily_df.loc[dateIdx]['balance'] = np.sum(daily_df.loc[dateIdx])
        daily_df.loc[dateIdx]['credit'] = account_df.loc[baseDate]['cum_credit']

    daily_df['total'] = daily_df['cash'] + daily_df['balance']
    daily_df['diff'] = daily_df['total'] - daily_df['credit']
    daily_df['return'] = daily_df['total'] / daily_df['credit'] -1

    daily_df = daily_df.dropna()

    return daily_df[['credit', 'cash', 'balance', 'total', 'diff', 'return']]


def visualize(fullDF, start_day, end_day):
    kospi_df = pdr.get_data_yahoo("^KS11", initDate, today)['Adj Close']

    # 일 수익률
    daily_ret = kospi_df.pct_change()

    # 누적수익률
    cum_ret = (1 + daily_ret).cumprod() - 1

    plt.figure(figsize=(16, 9))
    ax = plt.subplot()

    plt.plot(fullDF.index, fullDF['return'], color='red', label='JinYoung')
    plt.plot(cum_ret.index, cum_ret, color='blue', label='KOSPI')

    plt.legend(loc='best')
    plt.ylabel('Price')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))

    plt.show()


if __name__ == '__main__':

    today = date.today()

    # ----- 최초 실행 이후 (이진영)----
    # balance = 0  # 추가 투자금액
    # investAmt = 3_000_000  # 실제 투자 금액
    #
    # isRebalance = True
    # isCheck = True
    # initDate = date(2021, 11, 18)
    # dirPath = f"./Profile/JYLEE"

    # ----- 최초 실행 이후 (박종민)----
    balance = 0  # 추가 투자금액
    investAmt = 1_500_000  # 실제 투자 금액

    isRebalance = True
    isCheck = True
    initDate = date(2021, 12, 7)
    dirPath = f"./Profile/JMPARK"

    # ----- 최초 실행 ----
    # balance = 1_500_000  # 초기투자금액
    # investAmt = 1_500_000  # 실제 투자 금액
    # isRebalance = False
    # isCheck = False
    # initDate = today
    # dirPath = f"./{initDate}"

    if not isRebalance:
        dirPath = createFolder(today)

    # 전체 ETF 종목
    allETFs = ClassifyETF.getAllETFs(dirPath, True)

    # 1차 필터링 및 국내/해외, 주식/그외 로 분류
    # > 시간이 오래 걸림. 그룹별 파일이 존재할 경우 다시 실행할 때는 SKIP
    ClassifyETF.selectAndClassify(dirPath, initDate, True)

    # 2차 필터링 : 순위화하여 상위 30개 선택 ...
    targetETFs = []
    if not isRebalance:
        targetETFs.extend(TargetETF.selectTarget(dirPath))
    else:
        targetETFs.extend(getTargetETFs(dirPath))

    # 대상 30개에 대한 시세 데이터 수집 ...
    # > 시간이 오래 걸림. 파일이 존재할 경우 다시 실행할 때는 SKIP
    print('-' * 40)
    print('네이버 파이낸스에서 시세데이터를 크롤링합니다 ...')
    print('-' * 40)

    # commStartDate = TargetETF.crawlingTarget(targetETFs, dirPath, True)
    commStartDate = TargetETF.crawlingTarget_v1(targetETFs)

    print('-' * 40)
    print('공통 시작일자 :', commStartDate)
    print('-' * 40)

    weight = 1 / len(targetETFs)
    for asset in targetETFs:
        intAssetCode = int(asset)
        strAssetName = allETFs[allETFs['단축코드'] == intAssetCode]['한글종목약명'].values
        # print(f'\'{asset}\': {weight:.2f}, # {assetName[0]} https://finance.naver.com/item/main.naver?code={asset}')
        print(f'\'{asset}\': {strAssetName[0]} https://finance.naver.com/item/main.naver?code={asset}')

    # startDate = today - relativedelta(years=1)
    startDate = today - relativedelta(months=3)
    endDate = today - relativedelta(days=1)

    portfolio = LassoPortfolio(targetETFs, startDate, endDate)
    rsDict = portfolio.run()

    # < 0.05 이하의 데이터는 제외하고 weight 재산정 ...
    lstWeight = np.array(list(rsDict.values()))

    lstWeight = np.where(lstWeight <= 0.05, 0., lstWeight)
    if lstWeight.sum() != 0:
        lstWeight = lstWeight / lstWeight.sum()

    for idx, code in enumerate(rsDict.keys()):
        rsDict[code] = round(lstWeight[idx], 2)

    idxInitDay = pd.to_datetime(initDate)
    idxToday = pd.to_datetime(today)

    # 전날 종가 데이터를 가져온다.
    priceDF = getPriceData(targetETFs, idxInitDay, idxToday, dirPath, isRebalance)
    # print_full( priceDF )
    priceDF.to_csv(f"{dirPath}/PRC_CLOSE.csv")

    # 투자 데이터 생성 ...
    investDF, weightDF, runningBalance = invest(idxToday, balance, rsDict, priceDF, dirPath, isRebalance, isCheck)

    print('-' * 40)
    print('종목별 가중치 ... ')
    print('-' * 40)
    print_full(weightDF)

    print('-' * 40)
    print('종목별 실제 주식수 ... ')
    print('-' * 40)
    print_full(investDF)
    print('-' * 100)

    if isRebalance:
        TradingSignal.checkTradingSignal(weightDF, priceDF, dirPath)
        print('-' * 100)

    returnRatio = (runningBalance / investAmt) * 100 - 100
    print(f'현재 수익률 : {returnRatio:.2f}, (평가금액/투자금액: {runningBalance:,.0f}/{investAmt:,})')
    print('-' * 100)

    ### 최종 결과 출력 ...
    nTotalCnt = len(weightDF)

    lastWeight = weightDF.iloc[-1:]
    lastInvest = investDF.iloc[-1:]

    if nTotalCnt <= 1:  # 최초 출력인 경우 ...

        for assetCode in weightDF.columns:

            assetWeight = lastWeight[assetCode].values[0]
            if assetWeight <= 0:
                continue

            curPrice = priceDF.iloc[-1:][assetCode].values[0]

            intAssetCode = int(assetCode)
            investAmt = int(runningBalance * assetWeight)
            investCnt = lastInvest[assetCode].values[0]

            strAssetName = allETFs[allETFs['단축코드'] == intAssetCode]['한글종목약명'].values

            print(
                f"{curPrice:10,}, {investCnt:5d}, {investAmt:12,}, {assetWeight:.2f}"
                f"      # {assetCode} : {strAssetName[0]:25s}")

    else:
        prevIndex = weightDF.index[nTotalCnt - 2]

        prevWeight = weightDF.loc[prevIndex]
        prevInvest = investDF.loc[prevIndex]

        for assetCode in weightDF.columns:

            indSignal = 0

            prevAssetWeight = prevWeight[assetCode]
            lastAssetWeight = lastWeight[assetCode].values[0]

            if lastAssetWeight <= 0 and prevAssetWeight <= 0:
                continue

            intAssetCode = int(assetCode)
            strAssetName = allETFs[allETFs['단축코드'] == intAssetCode]['한글종목약명'].values

            curPrice = priceDF.iloc[-1:][assetCode].values[0]

            prevInvestCnt = prevInvest[assetCode]
            lastInvestCnt = lastInvest[assetCode].values[0]

            prevInvestAmt = int( curPrice * prevInvestCnt)
            lastInvestAmt = int(runningBalance * lastAssetWeight)

            # 없던 종목이 새롭게 추가된 경우
            if lastAssetWeight > 0 and prevAssetWeight <= 0:
                print( f"(+) {curPrice:10,.0f}, {lastInvestCnt:5.0f}({prevInvestCnt:5.0f}), {lastInvestAmt:12,}"
                       f", {lastAssetWeight:.2f}      # {assetCode} : {strAssetName[0]:25s}")

            # 기존에 있던 종목이 없어지는 경우
            elif lastAssetWeight <= 0 and prevAssetWeight > 0:
                print(f"(-) {curPrice:10,.0f}, {prevInvestCnt:5.0f}({prevInvestCnt:5.0f}), {prevInvestAmt:12,}"
                      f", {lastAssetWeight:.2f}      # {assetCode} : {strAssetName[0]:25s}")

            # 기존에 있던 종목의 wegith 가 바뀌는 경우
            else:
                print(f"(*) {curPrice:10,.0f}, {lastInvestCnt:5.0f}({prevInvestCnt:5.0f}), {lastInvestAmt:12,}"
                      f", {lastAssetWeight:.2f}      # {assetCode} : {strAssetName[0]:25s}")

    fullDF = calcFullData(investDF, priceDF, balance, isRebalance, isCheck, dirPath)

    # print_full( fullDF )
    visualize( fullDF, initDate, today )