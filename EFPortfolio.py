from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize


class EFPortfolio:
    def __init__(self, lstMember, startDate, endDate):
        self.lstMember = lstMember

        self.startDate = startDate
        self.endDate = endDate

        self.filePrefix = './Assets'

    # def __del__(self):
    #     print(f"... {self.startDate} ~ {self.endDate} ...")

    # 포트폴리오 구성 : 목표수익률(30% : _earning)를 만족하는 포트폴리오 중에 샤프지수가 최대가 되는 포트폴리오 선택
    def run(self, type="MAX_SHARPE"):
        lstMaxShape, lstMinRisk = self.efficientFrontier_v1()
        # lstMaxShape, lstMinRisk = self.efficientFrontier_v2()

        rsDict = {}
        for idx, code in enumerate(self.lstMember):
            if type == "MAX_SHARPE":
                rsDict[code] = round(lstMaxShape[idx], 2)
            elif type == "MIN_RISK":
                rsDict[code] = round(lstMinRisk[idx], 2)

        return rsDict

    def efficientFrontier_v1(self):
        masterDF = pd.DataFrame()
        for code in self.lstMember:
            masterDF[code] = self.getClosePrice(code)

        lenDates = len(masterDF)
        np.random.seed(lenDates)

        # print(masterDF)
        daily_ret = np.log(masterDF / masterDF.shift(1))
        # daily_ret = masterDF.pct_change()
        daily_cov = daily_ret.cov()

        yearly_ret = daily_ret.mean() * lenDates
        yearly_cov = daily_cov * lenDates

        num_ports = 25000
        num_stocks = len(self.lstMember)

        port_weights = np.zeros((num_ports, num_stocks))
        port_return = np.zeros(num_ports)
        port_risk = np.zeros(num_ports)
        port_sharpe = np.zeros(num_ports)

        for port in range(num_ports):
            weights = np.random.random(num_stocks)
            weights = weights / np.sum(weights)

            port_weights[port, :] = weights
            port_return[port] = np.dot(weights, yearly_ret)

            port_risk[port] = np.sqrt(np.dot(weights.T, np.dot(yearly_cov, weights)))
            port_sharpe[port] = port_return[port] / port_risk[port]

        rsDict = {'Returns': port_return, 'Risk': port_risk, 'Sharpe': port_sharpe}

        for idx, code in enumerate(self.lstMember):
            rsDict[code] = [w[idx] for w in port_weights]

        portfolioDF = pd.DataFrame(rsDict)

        max_sharpe = portfolioDF.loc[portfolioDF['Sharpe'] == portfolioDF['Sharpe'].max()]
        min_risk = portfolioDF.loc[portfolioDF['Risk'] == portfolioDF['Risk'].min()]

        maxSharpeList = max_sharpe.values.tolist()[0]
        minRiskList = min_risk.values.tolist()[0]

        # portfolioDF.plot.scatter(x='Risk', y='Returns', c='Sharpe', cmap='viridis',
        #                 edgecolors='k', figsize=(11, 7), grid=True)
        # plt.scatter(x=max_sharpe['Risk'], y=max_sharpe['Returns'], c='r',
        #             marker='*', s=300)
        # plt.scatter(x=min_risk['Risk'], y=min_risk['Returns'], c='r',
        #             marker='X', s=200)
        #
        # plt.title('Portfolio Optimization')
        # plt.xlabel('Risk')
        # plt.ylabel('Returns')
        # plt.show()

        return maxSharpeList[3:], minRiskList[3:]

    def efficientFrontier_v2(self):

        masterDF = pd.DataFrame()
        for code in self.lstMember:
            masterDF[code] = self.getClosePrice(code)

        lenDates = len(masterDF)
        np.random.seed(lenDates)

        daily_ret = np.log(masterDF / masterDF.shift(1))
        # daily_ret = masterDF.pct_change()
        daily_cov = daily_ret.cov()

        yearly_ret = daily_ret.mean() * lenDates
        yearly_cov = daily_cov * lenDates

        num_stocks = len(self.lstMember)

        def portfolio_return(weights):
            return np.sum(yearly_ret * weights)

        def portfolio_risk(weights):
            return np.sqrt(np.dot(weights.T, np.dot(yearly_cov, weights)))

        def neg_sharpe_ratio(weights):
            p_ret = np.sum(yearly_ret * weights)
            p_risk = np.sqrt(np.dot(weights.T, np.dot(yearly_cov, weights)))
            return - (p_ret / p_risk)

        # 샤프지수 최대화
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for asset in range(num_stocks))

        maxSharpe = minimize(neg_sharpe_ratio, num_stocks * [1. / num_stocks, ],
                             method='SLSQP', bounds=bounds, constraints=constraints)

        maxSharpe_return = portfolio_return(maxSharpe['x'])
        maxSharpe_risk = portfolio_risk(maxSharpe['x'])

        # print('MAX Sharpe :', maxSharpe['x'].round(2), f'return : {maxSharpe_return:.2f}, risk : {maxSharpe_risk:.2f}')

        # Risk(변동성) 최소화
        minRisk = minimize(portfolio_risk, num_stocks * [1. / num_stocks, ],
                           method='SLSQP', bounds=bounds, constraints=constraints)

        minRisk_return = portfolio_return(minRisk['x'])
        minRisk_risk = portfolio_risk(minRisk['x'])

        # print('MIN Risk :', minRisk['x'].round(2), f'return : {minRisk_return:.2f}, risk : {minRisk_risk:.2f}')

        # 지정된 수익률에 대한 프로파일을 리턴한다.
        constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - ret},
                       {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

        # 전체 데이터 계산하여 출력 ...
        ef_returns = np.linspace(minRisk_return, maxSharpe_return, 200)

        ef_portfolio = []
        for ret in ef_returns:
            result = minimize(portfolio_risk, num_stocks * [1. / num_stocks, ], method='SLSQP', bounds=bounds,
                              constraints=constraints)
            ef_portfolio.append(result)

        # plt.plot([p['fun'] for p in ef_portfolio], ef_returns, linestyle='-.', color='black',
        #          label='efficient frontier')
        # plt.scatter(x=maxSharpe_risk, y=maxSharpe_return, c='r', marker='*', s=300)
        # plt.scatter(x=minRisk_risk, y=minRisk_return, c='r', marker='X', s=200)
        #
        # plt.title('Portfolio Optimization')
        # plt.xlabel('Risk')
        # plt.ylabel('Returns')
        #
        # plt.show()

        return ef_portfolio[-1]['x'], ef_portfolio[0]['x']

    def getClosePrice(self, code):
        # print(f"./data/{self.filePrefix}_{code}.csv")

        df = pd.read_csv(f"./{self.filePrefix}/PRC_{code}.KS.csv", index_col=0)
        df.index = pd.to_datetime(df.index)

        # df = df.rename({'Adj Close': code}, axis='columns')
        filterDF = df.loc[self.startDate:self.endDate]

        return filterDF['close']


if __name__ == '__main__':
    today = date(2021, 11, 19)
    startDate = today - relativedelta(years=1)
    endDate = today - relativedelta(days=1)

    filePrefix = '2021-11-22'
    lstMember = ['102110', '329750', '153130', '360200', '360750', '278540', '261220', '069500', '157450', '133690']

    portfolio = EFPortfolio(lstMember, startDate, endDate, filePrefix)

    print(40 * '-')
    # rsDict = portfolio.run('MAX_SHARPE')
    rsDict = portfolio.run('MIN_RISK')

    print(rsDict)
