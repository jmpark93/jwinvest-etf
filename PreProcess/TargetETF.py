import math
import os
from datetime import date

import pandas as pd

from PreProcess.CrawlingNaver import CrawlingNaver


def loadData(fileName):
    df = pd.read_csv(f"{fileName}.csv", index_col=0)

    # print(f'{fileName} : ', len(df))
    return df


# [ 거래량 (+), 시가총액(+), 보수(-), 상장좌수(+) ]
def selectData(etf_df, limit):
    columns = ['단축코드', '거래량', '시가총액', '보수', '상장좌수', 'Score']
    ranking_df = pd.DataFrame(etf_df.index, columns=columns)

    ranking_df['단축코드'] = etf_df['단축코드']

    ranking_df['거래량'] = etf_df['거래량'].rank(ascending=True)
    ranking_df['시가총액'] = etf_df['시가총액'].rank(ascending=True)
    ranking_df['보수'] = etf_df['보수'].rank(ascending=False)
    ranking_df['상장좌수'] = etf_df['상장좌수'].rank(ascending=True)

    ranking_df['Score'] = ranking_df['거래량'] * 0.25 + \
                          ranking_df['시가총액'] * 0.25 + \
                          ranking_df['보수'] * 0.25 + \
                          ranking_df['상장좌수'] * 0.25

    ranking_df = ranking_df.sort_values(by='Score', ascending=False)

    # print_full(ranking_df)

    lstIntAssetCode = ranking_df.head(limit)['단축코드'].tolist()
    lstStrAssetCode = [f'{x:06d}' for x in lstIntAssetCode]

    return lstStrAssetCode


def selectTarget(dirPath):
    ## Loading Files ...
    domesticStock_df = loadData(f"{dirPath}/ETF_DomesticStock")
    domesticEtc_df = loadData(f"{dirPath}/ETF_DomesticEtc")
    overseaStock_df = loadData(f"{dirPath}/ETF_OverseaStock")
    overseaEtc_df = loadData(f"{dirPath}/ETF_OverseaEtc")

    ## 후보군 선정 ... 총 15개 : 7 + 3 + 7 + 3
    candidates = []
    candidates.extend(selectData(domesticStock_df, 7))
    candidates.extend(selectData(domesticEtc_df, 3))
    candidates.extend(selectData(overseaStock_df, 7))
    candidates.extend(selectData(overseaEtc_df, 3))

    return candidates


def crawlingTarget(candidates, dirPath, isFromFile):
    cspObj = CrawlingNaver()

    today = date.today()

    todayTimestamp = pd.Timestamp(f'{today.year}/{today.month}/{today.day}')
    comm_start_date = pd.Timestamp('2000/1/1 00:00')

    for asset in candidates:
        df = None
        filePath = f"ASSET_PRICE/PRC_{asset}.KS.csv"

        print( filePath )

        if os.path.isfile(filePath):
            print( 'File exist ... ' )
        else:
            print( 'File not exist ... ')

        if isFromFile:
            df = pd.read_csv(f"{dirPath}/PRC_{asset}.KS.csv", index_col=0)
            df.index = pd.to_datetime(df.index)

            if todayTimestamp not in df.index:
                crawlDF = cspObj.run(asset, 5)  # 대략 2달 정도 ...
                df = df.combine_first(crawlDF)
                df.update( crawlDF )
            else:
                crawlDF = cspObj.run(asset, 1)  # 대략 2 week 정도 ...
                df = df.combine_first(crawlDF)
                df.update( crawlDF )

        else:
            # assetName = f'{asset:06d}'
            df = cspObj.run(asset, 100)  # 4-year (1년 : 25 페이지)되도록 많은 데이터를 가져오도록 한다.

        head = df.head(1).to_csv(header=False).splitlines()
        tail = df.tail(1).to_csv(header=False).splitlines()

        df.index = pd.to_datetime(df.index)
        # print( df.index.min() )

        if df.index.min() > comm_start_date:
            comm_start_date = df.index.min()

        print(asset, '-->', head[0].split(",")[0], tail[0].split(",")[0])

        # print(assetDF.head(1).to_csv(header=False).splitlines())
        # print(assetDF.tail(1).to_csv(header=False).splitlines())

        if isFromFile:
            df.to_csv(f"{dirPath}/PRC_{asset}.KS.csv")
        else:
            assetName = f'{asset:06d}'
            df.to_csv(f"{dirPath}/PRC_{assetName}.KS.csv")

    return comm_start_date


def crawlingTarget_v1(candidates, dirPath='.'):
    cspObj = CrawlingNaver()

    today = date.today()

    todayTimestamp = pd.Timestamp(f'{today.year}/{today.month}/{today.day}')
    comm_start_date = pd.Timestamp('2000/1/1 00:00')

    for asset in candidates:
        df = None
        filePath = f"{dirPath}/Assets/PRC_{asset}.KS.csv"

        if os.path.isfile(filePath):
            df = pd.read_csv(filePath, index_col=0)
            df.index = pd.to_datetime(df.index)

            diff_days = (todayTimestamp - df.index[-1]).days + 1
            page_count = math.ceil( diff_days / 10 )

            crawlDF = cspObj.run(asset, page_count)
            df = df.combine_first(crawlDF)
            df.update(crawlDF)

        else:
            df = cspObj.run(asset, 100)  # 4-year (1년 : 25 페이지)되도록 많은 데이터를 가져오도록 한다.

        head = df.head(1).to_csv(header=False).splitlines()
        tail = df.tail(1).to_csv(header=False).splitlines()

        df.index = pd.to_datetime(df.index)

        if df.index.min() > comm_start_date:
            comm_start_date = df.index.min()

        print(asset, '-->', head[0].split(",")[0], tail[0].split(",")[0])

        df.to_csv(filePath)

    return comm_start_date


if __name__ == '__main__':
    today = date.today()
    dirPath = f"./{today}"

    # targetETFs = selectTarget(dirPath)
    targetETFs = ['069500']
    print(targetETFs)

    # commStartDate = crawlingTarget(targetETFs, dirPath, True)
    commStartDate = crawlingTarget_v1(targetETFs, '..')
    print('공통 시작일자 :', commStartDate)

    weight = 1 / len(targetETFs)
    for asset in targetETFs:
        print(f'\'{asset}\': {weight:.2f}, #')
