import os
import re

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from Utility import print_full


def loadAsset(assetName, filePrefix):
    # print(f"./data/{filePrefix}_{assetName}.csv")
    assetDF = pd.read_csv(f"./data/{filePrefix}_{assetName}.csv", index_col=0)
    assetDF.index = pd.to_datetime(assetDF.index)

    return assetDF


def drawChart(assetDF):
    plt.figure(figsize=(10, 4))
    ax = plt.subplot()

    plt.plot(assetDF.index, assetDF['close'])
    plt.xlabel('')
    plt.ylabel('Close Price')
    # plt.tick_params(
    #     axis='x',  # changes apply to the x-axis
    #     which='both',  # both major and minor ticks are affected
    #     bottom=False,  # ticks along the bottom edge are off
    #     top=False,  # ticks along the top edge are off
    #     labelbottom=False)  # labels along the bottom edge are off
    plt.show()


def drawTimeSeriesChart(assetDF):
    fig = px.line(assetDF, y='close', title='Close Prcie Time Series')

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.show()


def drawAllTimeSeriesChart(assetDict):
    fig = go.Figure()

    for key in assetDict.keys():
        fig.add_trace(
            go.Scatter(x=assetDict[key].index, y=assetDict[key]['close'], mode='lines', name=key))

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=12, label="1y", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.show()


def drawMAChart(assetDF):
    short_window = 50
    long_window = 100

    assetMA = assetDF.drop(columns=['high', 'low', 'open', 'volume'])

    assetMA['MA50'] = assetMA['close'].rolling(window=short_window).mean()
    assetMA['MA100'] = assetMA['close'].rolling(window=long_window).mean()

    assetMA['Signal'] = 0.0
    assetMA['Signal'][short_window:] = np.where(
        assetMA['MA50'][short_window:] > assetMA['MA100'][short_window:], 1.0, 0.0)

    assetMA['Entry/Exit'] = assetMA['Signal'].diff()
    # print_full( assetMA )

    plt.figure(figsize=(16, 9))
    ax = plt.subplot()

    plt.plot(assetMA['close'].index, assetMA['close'], color='lightgray', label='Price(Close)')
    plt.plot(assetMA['close'].index, assetMA['MA50'], label='50-days SMA')
    plt.plot(assetMA['close'].index, assetMA['MA100'], label='100-days SMA')

    plt.scatter(assetMA[assetMA['Entry/Exit'] == 1.0].index, assetMA[assetMA['Entry/Exit'] == 1.0]['close'],
                color='red', marker='*', s=300)

    plt.scatter(assetMA[assetMA['Entry/Exit'] == -1.0].index, assetMA[assetMA['Entry/Exit'] == -1.0]['close'],
                color='blue', marker='*', s=300)

    plt.legend(loc='best')
    plt.ylabel('Price')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))

    plt.show()

## 지난 데이터가 없어서 일단 제외한 종목들 ...
# 360750.KS 2020-08-07 2021-11-05
# 360200.KS 2020-08-07 2021-11-05
# 367380.KS 2020-10-29 2021-11-05
# 310970.KS 2018-11-20 2021-11-05
# 314250.KS 2019-01-10 2021-11-05
# 329650.KS 2019-07-04 2021-11-05
# 329750.KS 2019-07-24 2021-11-05

def checkDate():
    global member, head, tail
    ## 데이터의 시작, 종료날짜 ...
    lstAssetDF = {}
    comm_start_date = pd.Timestamp('2000/1/1 00:00')
    for member in lstMember:
        lstAssetDF[member] = loadAsset(member, filePrefix)

        head = lstAssetDF[member].head(1).to_csv(header=False).splitlines()
        tail = lstAssetDF[member].tail(1).to_csv(header=False).splitlines()

        # print( df.index.min() )
        if lstAssetDF[member].index.min() > comm_start_date:
            comm_start_date = lstAssetDF[member].index.min()

        print(member, head[0].split(",")[0], tail[0].split(",")[0])
    print('시작일자 :', comm_start_date)


if __name__ == '__main__':

    filePrefix = 'Profile/JYLEE'

    path = f'./{filePrefix}/'
    files = os.listdir(path)

    ## 현재 디렉토리에 존재하는 데이터 목록 생성
    lstMember = []
    for f in files:
        member = re.findall(f'PRC_(.+).KS.csv', f)
        if len(member) != 0:
            lstMember.extend( member )

    print( lstMember )

    ## 동일 Weight로 할당하여 출력
    weight = 1 / len(lstMember)
    for asset in lstMember:
        print(f'\'{asset}\': {weight:.2f},')

    ## 데이터 시작 날짜 ~ 마지막 날짜 체크 ...
    # checkDate()

    ## 전체종목 출력
    # drawAllTimeSeriesChart(lstAssetDF)

    ## 단일종목 출력
    # drawTimeSeriesChart(lstAssetDF['367380.KS'])
    # drawChart(lstAssetDF['367380.KS'])
    # drawMAChart(lstAssetDF['367380.KS'])
