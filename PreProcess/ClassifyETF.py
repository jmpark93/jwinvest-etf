import json
from datetime import date
from random import randint
from time import sleep

import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta


def getAllETFs(directory, isFromFile):

    if isFromFile:
        df = pd.read_csv(f"{directory}/ETF_ALL.csv", index_col=0)
        return df

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36',
        'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201030104'
    }

    otp_data = {
        'share': '1',
        'csvxls_isNo': 'false',
        'name': 'fileDown',
        'url': 'dbms/MDC/STAT/standard/MDCSTAT04601'
    }
    otp_url = 'http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd'

    otp_response = requests.post(url=otp_url, headers=headers, data=otp_data)
    # print( otp_response.content )

    download_data = {
        'code': otp_response.content  # 위에서 획득한 OTP를 여기 넣어주자
    }

    download_url = 'http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'

    download_response = requests.post(url=download_url, headers=headers, data=download_data)

    # print( download_response.text )
    # print( chardet.detect(download_response.content) )
    utf8_contents = download_response.content.decode('euc-kr').encode('utf-8')

    with open(f"{directory}/ETF_ALL.csv", 'wb') as f:
        f.write(utf8_contents)

    df = pd.read_csv(f"{directory}/ETF_ALL.csv", index_col=0)
    return df


def filtering(df, base_day, period=1):
    # base_day = date(2021, 11, 3)
    oneyear_day = base_day - relativedelta(years=period)

    # 상장일 : 1년 이상 경과한 주식만 ...
    df['상장일'] = pd.to_datetime(df['상장일'])
    filter01_df = df[df['상장일'] < pd.to_datetime(oneyear_day)]
    print('1차 필터링 (1년 이하 제외) : ', len(filter01_df))

    # 인버스, 레버리지 제거 ... '일반'만 선택
    filter02_df = filter01_df[filter01_df['추적배수'] == '일반 (1)']
    print('2차 필터링 (일반 이외 제외) : ', len(filter02_df))

    # 상장좌수 1,000,000 만건 이하 제거
    filter03_df = filter02_df[filter02_df['상장좌수'] > 1000000]
    print('3차 필터링 (상장좌수 > 1000000 일백만건) : ', len(filter03_df))

    domestic_df = filter03_df[filter03_df['기초시장분류'] == '국내']
    oversea_df = filter03_df[filter03_df['기초시장분류'].isin(['국내&해외', '해외'])]

    return domestic_df, oversea_df


def getETFDetail(isuCd):
    etf_headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36',
        'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201030105'
    }

    etf_data = {
        'bld': 'dbms/MDC/STAT/standard/MDCSTAT04701',
        'isuCd': isuCd,
    }
    # dbms/MDC/STAT/standard/MDCSTAT04704
    # etf_data = {
    #     'bld': 'dbms/MDC/STAT/standard/MDCSTAT04701',
    #     'tboxisuCd_finder_secuprodisu1_0': '114100/KBSTAR 국고채3년',
    #     'isuCd': isuCd,
    #     'isuCd2': '',
    #     'codeNmisuCd_finder_secuprodisu1_0': 'KBSTAR 국고채3년',
    #     'param1isuCd_finder_secuprodisu1_0': '',
    #     'csvxls_isNo': 'false'
    # }

    etf_url = 'http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd'

    etf_response = requests.post(url=etf_url, headers=etf_headers, data=etf_data)
    # print(isuCd, etf_response)
    etf_contents = None
    try:
        etf_contents = etf_response.json()
        # print( etf_contents )
    except json.decoder.JSONDecodeError as errmsg:
        print(f'Error in {isuCd} ... {errmsg}')

    # [ 시가, 거래량, 시가총액, 보수 ]
    eft_list = [np.NaN] * 4

    eft_list[0] = float(etf_contents['TDD_CLSPRC'].replace(',', ''))
    eft_list[1] = float(etf_contents['ACC_TRDVOL'].replace(',', ''))
    eft_list[2] = float(etf_contents['MKTCAP'].replace(',', ''))
    eft_list[3] = float(etf_contents['ETF_TOT_FEE'])

    return eft_list


def addAdditionInfo(etf_df):
    columns = ['시가', '거래량', '시가총액', '보수']
    df = pd.DataFrame(index=etf_df.index, columns=columns)

    for idx, isuCd in enumerate(etf_df.index):
        rsDetail = getETFDetail(isuCd)
        df.loc[isuCd] = rsDetail

        # 가끔씩 차단되는 듯 함 : 403 Forbidden ...
        rand_value = randint(1, 4)
        sleep(rand_value)
        print('\r', idx + 1, '/', len(etf_df), isuCd, end='')

    merge_df = pd.concat([etf_df, df], axis=1)

    return merge_df


def saveFilterDF(fileName, etf_df):
    etf_df = etf_df.drop(
        ['한글종목명', '한글종목약명', '영문종목명', '기초지수명', '지수산출기관', '추적배수', '복제방법', '기초시장분류',
         '기초자산분류', '운용사', 'CU수량', '과세유형'], axis=1)

    # print_full(etf_df)

    etf_df.to_csv(f"{fileName}.csv")


def selectAndClassify(dirPath, startDate, isFromFile):
    # ['단축코드', '한글종목명', '한글종목약명', '영문종목명', '상장일', '기초지수명', '지수산출기관', '추적배수',
    #  '복제방법', '기초시장분류', '기초자산분류', '상장좌수', '운용사', 'CU수량', '총보수', '과세유형']

    if isFromFile:
        return

    df = pd.read_csv(f"{dirPath}/ETF_ALL.csv", index_col=0)
    print('전체 자료 개수 : ', len(df))
    print('-' * 40)

    domestic_df, oversea_df = filtering(df, startDate)

    print('-' * 40)
    ## 추가 데이터 확보 ... ['시가', '거래량', '시가총액', '보수']
    print("... 국내시장 : 추가 데이터 확보 중(시간이 많이 걸립니다) ... ")
    domestic_df = addAdditionInfo(domestic_df)

    print()
    print("... 해외시장 : 추가 데이터 확보 중(시간이 많이 걸립니다) ... ")
    oversea_df = addAdditionInfo(oversea_df)

    # 10개 선택 : 국내 주식(111)
    #  5개 선택 : 그 외(21: '채권' '기타' '혼합자산' '부동산' '통화')
    domesticStock_df = domestic_df[domestic_df['기초자산분류'] == '주식']
    domesticEtc_df = domestic_df[domestic_df['기초자산분류'] != '주식']

    print()
    print('-' * 40)
    print(f'국내시장 전체(주식/그외) : {len(domestic_df)}({len(domesticStock_df)}/{len(domesticEtc_df)})')

    # 10개 선택 : 해외 주식(43)
    #  5개 선택 : 그 외(19: '원자재' '채권' '부동산' '혼합자산')
    overseaStock_df = oversea_df[oversea_df['기초자산분류'] == '주식']
    overseaEtc_df = oversea_df[oversea_df['기초자산분류'] != '주식']

    print(f'해외시장(혼합포함) 전체(주식/그외) : {len(oversea_df)}({len(overseaStock_df)}/{len(overseaEtc_df)})')
    print('-' * 40)

    saveFilterDF(f"{dirPath}/ETF_DomesticStock", domesticStock_df)
    saveFilterDF(f"{dirPath}/ETF_DomesticEtc", domesticEtc_df)
    saveFilterDF(f"{dirPath}/ETF_OverseaStock", overseaStock_df)
    saveFilterDF(f"{dirPath}/ETF_OverseaEtc", overseaEtc_df)


if __name__ == '__main__':
    today = date.today()
    dirPath = f"./{today}"

    getAllETFs(dirPath)
    selectAndClassify(dirPath)
