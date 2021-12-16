import sys
from datetime import datetime, date

import pandas_datareader as pdr
from dateutil.relativedelta import relativedelta


def saveAsset(assetName, startDate, filePrefix):
    # print(f'{assetName}', 50 * '-')

    assetDF = pdr.get_data_yahoo(assetName, startDate, filePrefix)

    head = assetDF.head(1).to_csv(header=False).splitlines()
    tail = assetDF.tail(1).to_csv(header=False).splitlines()

    print(assetName, head[0].split(",")[0], tail[0].split(",")[0])

    # print(assetDF.head(1).to_csv(header=False).splitlines())
    # print(assetDF.tail(1).to_csv(header=False).splitlines())

    assetDF.to_csv(f"./data/{filePrefix}_{assetName}.csv")


if __name__ == '__main__':

    # today = datetime.today().date()
    today = date(2021, 11, 3)

    startDate = today - relativedelta(years=4)
    endDate = today - relativedelta(days=1)

    print(f'데이터수집 : {startDate} ~ {endDate}\n')

    lstMember = [ '114100.KS',  # 국내채권, KBSTAR 국고채3년
                  '132030.KS',  # 금,     KODEX 골드선물(H)
                  '261240.KS',  # 달러,    KODEX 미국달러선물
                  '161510.KS',  # 배당,    ARIRANG 고배당주
                  '069500.KS']  # 국내주식, KODEX 200

    # lstMember = [
    #     '069500.KS',
    #     '102110.KS',
    #     '122630.KS',
    #     '278530.KS',
    #     '292150.KS',
    #     '130680.KS',
    #     '139260.KS',
    #     '261220.KS',
    #     '152100.KS',
    #     '117700.KS',
    #     '148020.KS',
    #     '091170.KS',
    #     '305720.KS',
    #     '233740.KS',
    #     '278540.KS',
    #     '139220.KS',
    #     '161510.KS',
    #     '144600.KS',
    #     '228800.KS',
    #     '305540.KS',
    #     '105190.KS',
    #     '139230.KS',
    #     '091180.KS',
    #     '295040.KS',
    #     '251350.KS',
    #     '252710.KS'
    # ]

    # lstMember = ['364980.KS',
    #              '360750.KS',
    #              '360200.KS']

    for member in lstMember:
        saveAsset(member, startDate, endDate)
