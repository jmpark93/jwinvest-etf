import os
from datetime import date

from PreProcess import ClassifyETF, TargetETF


def createFolder(directory):
    current_path = os.getcwd()
    mkdir_path = f"{current_path}/{directory}"

    try:
        if not os.path.exists(mkdir_path):
            os.makedirs(mkdir_path)
    except OSError:
        print('Error: Creating directory. ' + mkdir_path)

    return mkdir_path


if __name__ == '__main__':

    today = date.today()
    dirPath = createFolder(today)

    # 전체 ETF 종목
    allETFs = ClassifyETF.getAllETFs(dirPath)

    # 1차 필터링 및 국내/해외, 주식/그외 로 분류
    # > 시간이 오래 걸림. 파일이 존재할 경우 다시 실행할 때는 SKIP
    ClassifyETF.selectAndClassify(dirPath)

    # 2차 필터링 : 순위화하여 상위 30개 선택 ...
    targetETFs = TargetETF.selectTarget(dirPath)
    # print( targetETFs )

    # 대상 30개에 대한 시세 데이터 수집 ...
    # > 시간이 오래 걸림. 파일이 존재할 경우 다시 실행할 때는 SKIP
    commStartDate = TargetETF.crawlingTarget(targetETFs, dirPath)

    print('-' * 40)
    print('공통 시작일자 :', commStartDate)
    print('-' * 40)

    weight = 1 / len(targetETFs)
    for asset in targetETFs:
        assetName = allETFs[allETFs['단축코드'] == asset]['한글종목약명'].values
        print(f'\'{asset:06d}\': {weight:.2f}, # {assetName[0]}, https://finance.naver.com/item/main.naver?code={asset:06d}')
