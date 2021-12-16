from datetime import date

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from dateutil.relativedelta import relativedelta
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RepeatedKFold

from Utility import print_full


class LassoPortfolio:
    def __init__(self, lstMember, startDate, endDate):
        self.lstMember = lstMember

        self.startDate = startDate
        self.endDate = endDate

        self.filePrefix = './Assets'

    def getClosePrice(self, code):
        # if len(code)
        df = pd.read_csv(f"{self.filePrefix}/PRC_{code}.KS.csv", index_col=0)
        # df = pd.read_csv(f"./data/{self.filePrefix}_{code}.csv", index_col=0)

        df.index = pd.to_datetime(df.index)

        # df = df.rename({'Adj Close': code}, axis='columns')
        filterDF = df.loc[self.startDate:self.endDate]

        return filterDF['close']

    def run(self):
        X, y = self.getTrainData()
        config = self.getConfig(X, y)

        model = Lasso(alpha=config['alpha'])
        model.fit(X, y)

        y_train_predict = model.predict(X)

        rmse = (np.sqrt(mean_squared_error(y, y_train_predict)))
        rmse = round(rmse, 2)
        score = round(model.score(X, y), 2)

        # print( config, f", RMSE: {rmse}, Score: {score}" )
        # print( model.coef_ )

        # 음수 값 --> 무조건 0으로 수정
        lstWeight = np.where(model.coef_ <= 0, 0., model.coef_ )

        if lstWeight.sum() != 0:
            weight = lstWeight / lstWeight.sum()
        else:
            weight = lstWeight

        # print( '초기 : ', weight )

        # lstWeight = np.where(lstWeight <= 0.05, 0., lstWeight )
        #
        # if lstWeight.sum() != 0:
        #     weight = lstWeight / lstWeight.sum()
        # else:
        #     weight = lstWeight
        #
        # print( '수정 : ', weight )

        rsDict = {}
        for idx, code in enumerate(self.lstMember):
            rsDict[code] = round(weight[idx], 2)

        return rsDict

        # plt.scatter(y, y_train_predict)
        # plt.xlabel("Actual")
        # plt.ylabel("Predicted")
        # plt.xticks(range(0, int(max(y)), 2))
        # plt.yticks(range(0, int(max(y)), 2))
        # plt.title("Actual vs Predicted")
        #
        # plt.show()

    def getTrainData(self):

        masterDF = pd.DataFrame()
        for code in self.lstMember:
            masterDF[code] = self.getClosePrice(code)

        # print_full( masterDF )

        # 일 수익률
        daily_ret = masterDF.pct_change()
        # 누적수익률
        cum_ret = (1 + daily_ret).cumprod() - 1

        # 결측치 확인 및 제거
        # print(masterDF.isnull().sum())
        # print(masterDF.isna().sum())
        cum_ret = cum_ret.dropna()

        # print_full( cum_ret )

        X = cum_ret.values
        y = np.linspace(0, 0.2, len(X))

        # print_full( y )
        return X, y

    def getConfig(self, X, y):
        model = Lasso( tol=1e-3 )

        # define model evaluation method
        # Repeated 5 Fold Cross Validation
        cv = RepeatedKFold(n_splits=4, n_repeats=3, random_state=1)

        # define grid
        grid = dict()
        # grid['alpha'] = np.arange(0.0001, 0.01, 0.0001)
        grid['alpha'] = np.arange(0.00005, 0.01, 0.0001)

        # define search
        search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
        # perform the search
        results = search.fit(X, y)

        # summarize
        # MAE(Mean Absolute Error : 평균절대오차)
        # print('MAE: %.3f' % results.best_score_)
        # print('Config: %s' % results.best_params_)

        return results.best_params_


if __name__ == '__main__':
    # 2019-10-26 ~ 2020-10-25
    # today = date(2020, 10, 26)
    today = date(2021, 11, 19)
    startDate = today - relativedelta(years=1)
    endDate = today - relativedelta(days=1)

    filePrefix = '2021-11-22'
    lstMember = ['102110', '329750', '153130', '360200', '360750', '278540', '261220', '069500', '157450', '133690']

    portfolio = LassoPortfolio(lstMember, startDate, endDate, filePrefix)

    rsDict = portfolio.run()

    print(rsDict)
    #
    # today = date(2021, 11, 19)
    # startDate = today - relativedelta(years=1)
    # endDate = today - relativedelta(days=1)
    #
    # filePrefix = '2021-11-22'
    # lstMember = ['102110', '329750', '153130', '360200', '360750', '278540', '261220', '069500', '157450', '133690']
    #
    # portfolio = EFPortfolio(lstMember, startDate, endDate, filePrefix)
    #
    # print(40 * '-')
    # # rsDict = portfolio.run('MAX_SHARPE')
    # rsDict = portfolio.run('MIN_RISK')
    #
    # print(rsDict)