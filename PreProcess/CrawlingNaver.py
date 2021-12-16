import os
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from Utility import print_full


class CrawlingNaver:
    def __init__(self):
        self.url = ""
        self.html = None
        self.lastPage = 25  # 1년 기준으로 ... 주말제외 평일 250일 정도 ... 1페이지 당 10일치 데이터

    def __del__(self):
        pass

    # 네이버에서 주식 시세를 읽어서 데이터프레임으로 반환
    def run(self, code, pagesCnt=25):
        """
        네이버에서 주식 시세를 읽어서 데이터프레임으로 반환
        pagesCnt 디폴트 값 : 25 (1년 기준 > 주말 제외 평일 250일 > 1페이지 당 10일치 데이터)
        @rtype: pandas.DataFrame
        """
        df = pd.DataFrame()

        try:
            self.url = f"http://finance.naver.com/item/sise_day.nhn?code={code}"
            self.html = BeautifulSoup(requests.get(self.url,
                                                   headers={'User-agent': 'Mozilla/5.0'}).text, "lxml")

            pgrr = self.html.find("td", class_="pgRR")
            if pgrr is None:
                return df

            s = str(pgrr.a["href"]).split('=')

            lastPage = pagesCnt
            if pagesCnt == -1:
                lastPage = int(s[-1])
            else:
                lastPage = min(int(s[-1]), pagesCnt)

            # print(lastPage)

            for page in range(1, lastPage + 1):
                pg_url = '{}&page={}'.format(self.url, page)
                df = df.append(pd.read_html(requests.get(pg_url,
                                                         headers={'User-agent': 'Mozilla/5.0'}).text)[0])

                tmnow = datetime.now().strftime('%Y-%m-%d %H:%M')
                # print('[{}] {} : {:04d}/{:04d} pages are downloading...'.
                #       format(tmnow, code, page, lastPage + 1), end="\r")

            df = df.rename(columns={'날짜': 'date', '종가': 'close', '전일비': 'diff'
                , '시가': 'open', '고가': 'high', '저가': 'low', '거래량': 'volume'})

            df['date'] = df['date'].replace('.', '-')
            df['date'] = pd.to_datetime( df['date'], format='%Y.%m.%d' )
            df = df.dropna()

            df[['close', 'diff', 'open', 'high', 'low', 'volume']] \
                = df[['close', 'diff', 'open', 'high', 'low', 'volume']].astype(int)

            df = df[['date', 'open', 'high', 'low', 'close', 'diff', 'volume']]

            df = df.set_index('date')
            df = df.sort_index(ascending=True)

        except Exception as e:
            print('예외 발생 :', str(e))
            return df

        return df


if __name__ == '__main__':
    cspObj = CrawlingNaver()

    # df = cspObj.run('032640', 1)
    df = cspObj.run('032640')
    print(type(df))
    print_full(df)
