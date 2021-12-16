## 환경
* Python 3.8.5
* 추가 라이브러리 : requirements.txt

## 실행 절차 
### 데이터 수집 
* CollectAsset.py 
* 주의 : 신규 ET 경우 분석하려는 기간내의 데이터가 없을 수 있으므로 데이터내의 날짜 확인 필요함   

### 수집된 데이터 시각화 
* VisualizeAsset.py

### 분석 
* Portfolio.py 
  * Markowitz Portfolio Optimization (Efficient Frontier)
  * 주어진 날짜에서 1년 전 훈련데이터로 Weight 산정 
  * Sharpe(최대), RISK(최소) 중 지정하여 사용 

* Lasso.py (구현 예정)
  * 목표 수익률을 지정하여 실행하면 될 것으로 보임
  * 목표 수익률 지정시 고정하여 훈련데이터 생성 
  * 또는, 날짜에 따라 0 ~ 목표수익률로 증가시켜 훈련데이터를 구성
  * 훈련데이터 보강 
    * ETF 수를 늘리고 그 중에서 목표수익률을 달성을 위한 대상 ETF가 선정되도록 ...   
    
### 백테스트 
* BackTest.py 
  * start_day ~ end_day : 정의된 자산분배 weight 값으로 백테스트 수행할 날짜 
  * initial_balance : 투자액 
  * rebalancing_date : 재분배 일자(월말, 월초, 지정날짜)
  * basePortfolio : 고정비율 자산분배 포트폴리오 지정 