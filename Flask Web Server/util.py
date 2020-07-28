import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def min_max_scailing(num, min_val, max_val):
    """[입력값을 정규화(0~1)값으로 바꿉니다.]

    Args:
        num ([type]): [description]
        min_val ([type]): [description]
        max_val ([type]): [description]

    Returns:
        [type]: [description]
    """

    min_max_value = (num-min_val) / (max_val-min_val)
    return min_max_value # return에는 수식 X

def pre_processing(gs, js):
    """[입력값을 전처리해서 변환합니다.]

    Args:
        gs ([type]): [description]
        js ([type]): [description]

    Returns:
        [type]: [description]
    """
    gs = gs * 100000000
    gs_MIN = 6236269000000
    gs_MAX = 14695530000000
    js_MIN = 88.8
    js_MAX = 114.8

    gs = min_max_scailing(gs, gs_MIN, gs_MAX)
    js = min_max_scailing(js, js_MIN, js_MAX)

    #print(gs, js) # min-max-scailing 완료
    return gs, js

def calc_sum_price(prev, pred):
    """[실적값과 예측값을 더한 sum값을 구합니다.]

    Args:
        prev ([type]): [실적값]
        pred ([type]): [예측값]

    Returns:
        [type]: [description]
    """
    prev = list(map(int, prev)) # 리스트의 각 원소를 str -> int
    pred = list(map(int, pred)) # 리스트의 각 원소를 str -> int
    prev_sum = sum(prev)
    pred_sum = sum(pred)
    sum_price = str(format(prev_sum + pred_sum, ",")) # 세자리 수 마다 컴마(,)
    
    return sum_price

def CAGR(gs, js, gs_percent, js_percent, remain_count):
    """[gs, js 각각의 CAGR 계산하기]

    Args:
        gs ([type]): [description]
        js ([type]): [description]
        gs_percent ([type]): [description]
        js_percent ([type]): [description]
        remain_count ([type]): [description]

    Returns:
        [type]: [description]
    """
    first_gs = gs
    first_js = js
    end_gs = gs * (1+ gs_percent/100)
    end_js = js * (1+ js_percent/100)
    CAGR_gs = (end_gs/first_gs)**(1/remain_count) - 1 # gs의 연평균 증가율 
    CAGR_js = (end_js/first_js)**(1/remain_count) - 1 # js의 연평균 증가율 

    return CAGR_gs, CAGR_js

def fill_price_lists(i, gs, js, lr, MAX_PRICE, MIN_PRICE, temp_price_list, price_list):
    """[temp_price_list와 price_list 제작]

    Args:
        i ([type]): [description]
        gs ([type]): [description]
        js ([type]): [description]
        lr ([type]): [description]
        MAX_PRICE ([type]): [description]
        MIN_PRICE ([type]): [description]
        temp_price_list ([type]): [description]
        price_list ([type]): [description]
    """
    X_test = pd.DataFrame( np.array([[gs, js]]) ) # index 마지막 위치의 gs,js값이 test데이터가 된다.
    Y_pred = lr.predict(X_test) # 예측하기 
    Y_pred = round(float(Y_pred),2)

    # 예측값 -> 원본값 추출해내기 
    price = Y_pred * (MAX_PRICE - MIN_PRICE) + MIN_PRICE
    
    temp_price_list.append(int(price/100000000))            # 컴마(,) 없는 버전
    price_list[i] = str(format(int(price/100000000), ","))  # 컴마(,) 있는 버전

    #return temp_price_list, price_list

def make_lists(start_date, end_date, df, count, feature_names):
    """[각종 list들을 생성하는 메서드]

    Args:
        start_date ([type]): [description]
        end_date ([type]): [description]
        df ([type]): [description]
        count ([type]): [총 개월 수]
        feature_names ([type]): [description]

    Returns:
        [type]: [description]
    """
    # 시작~끝 월(月) 리스트 만들기 (date_list)
    date_list = []

    start_year = start_date[:4] # 2020
    start_month = start_date[5:] # 01
    end_year = end_date[:4]
    end_month = end_date[5:]

    d_start = start_month+'/1/'+start_year
    d_end = end_month+'/1/'+end_year

    date_ = pd.date_range(start=d_start, end=d_end, freq='MS')

    for i in range(len(date_)):
        temp = str(date_[i])[:4] + '년' + str(date_[i])[5:7] + '월'
        date_list.append(temp)
    print('date_list = ', date_list)


    temp_start_date  = start_year + start_month # 202001
    LIMIT_DATE = end_year + end_month           # 202012

    STANDARD_MONTH = 5

    # 실적 값 가져올 '개월수' 구하기 
    prev_count = 0
    if int(temp_start_date) <= int(LIMIT_DATE):
        if int(start_year) == 2020:
            prev_count = STANDARD_MONTH - int(start_month) + 1
        elif int(start_year) < 2020:
            prev_year_count = 2020 - int(start_year)
            prev_month_count = 12 - int(start_month) + 1 + STANDARD_MONTH
            prev_count = (prev_year_count - 1)*12 + prev_month_count
        else:
            prev_count = 0
    print('count = ', count)
    if prev_count < 0: # 예외 처리 
        prev_count = 0
    print('prev_count = ', prev_count)
    

    # 실적 값 가져와서 prev_count만큼 리스트2개 생성 -> (temp_prev_price_list, prev_price_list)
    prev_price_list = [0]*prev_count
    start_index = len(df[feature_names]) - prev_count
    print('start_index = ', start_index)

    feature_names = ['real_is_price']

    # 실적 값을 리스트 각각에 넣기 
    temp_prev_price_list = []
    for i in range(prev_count):
        price = float(df[feature_names].iloc[start_index].values[0])
        temp_prev_price_list.append(int(price/100000000))               # 컴마(,) 없는 버전
        prev_price_list[i] = str(format(int(price/100000000), ","))     # 컴마(,) 있는 버전
        start_index += 1
    print('prev_price_list = ', prev_price_list)

    # 예측할 기간 만큼 리스트 2개 생성 -> (temp_price_list, price_list)
    remain_count = count - prev_count
    price_list = [0]*remain_count
    temp_price_list = []

    return date_list, prev_count, prev_price_list, temp_prev_price_list, \
                prev_price_list, remain_count, price_list, temp_price_list

def price_predict(gs_percent, js_percent, all_percent, option, start_date, end_date, count):
    """[ML학습을 수행해서 예측값을 추출해 냅니다.]

    Args:
        gs_percent ([type]): [description]
        js_percent ([type]): [description]
        all_percent ([type]): [description]
        option ([type]): [계정 코드]
        start_date ([type]): [description]
        end_date ([type]): [description]
        count ([type]): [총 개월 수]

    Returns:
        [type]: [description]
    """
    MAX_PRICE = 208375466507 
    MIN_PRICE = 65604000506 

    try:
        if option == '100':
            df = pd.read_csv("assemble_100.csv") # Code= 100 (전사 매출)
        elif option == '300':
            df = pd.read_csv("assemble_300.csv") # Code= 300 (국내 매출)
        else:
            pass
    except Exception as e:
        print('엑셀 파일을 읽어올 수 없습니다: \n', e)

    #df = df.dropna()
    df.dropna(inplace = True)

    feature_names = ['gs', 'js']
    X_train = df[feature_names]

    feature_names = ['is_price']
    Y_train = df[feature_names]

    # 모델 생성 
    lr = LinearRegression(fit_intercept=True, normalize=False, n_jobs=None)
    # 모델 학습 
    lr.fit(X_train, Y_train)

    
    date_list, prev_count, prev_price_list, temp_prev_price_list, \
        prev_price_list, remain_count, price_list, temp_price_list = make_lists(start_date, end_date, df, count, feature_names)



    gs = df['gs'].iloc[88] # 마지막 값으로 gs 업데이트(=이미 전처리 된 값)
    js = df['js'].iloc[88] # 마지막 값으로 js 업데이트(=이미 전처리 된 값)

    # 조건문 길어지면 밖으로 빼는 이유: 단위테스트에서 문제됨 
    CONDITION1 = (int(gs_percent) == 0 and int(js_percent) == 0 and int(all_percent) == 0)
    CONDITION2 = (int(gs_percent) == 0 and int(js_percent) == 0)
    CONDITION3 = (int(all_percent) == 0)

    # 입력 값이 모두 0일때, 예외 처리 
    if CONDITION1: 

        # 사용자가 입력한 월(月)만큼 while문 반복 수행
        i = 0
        while i < remain_count:

            fill_price_lists(i, gs, js, lr, MAX_PRICE, MIN_PRICE, temp_price_list, price_list)

            i += 1

        # sum구하기 
        sum_price = calc_sum_price(temp_prev_price_list, temp_price_list)

        # prev+pred 가격 리스트
        all_price_list = prev_price_list + price_list
        print('all_price_list = ', all_price_list)

        return prev_count, date_list, sum_price, all_price_list

    # 전월대비 증감(%)만 사용자가 입력했을 때, (핵심 로직)
    elif CONDITION2: 

        CAGR_gs, CAGR_js = CAGR(gs, js, all_percent, all_percent, remain_count)
        print('CAGR_gs, CAGR_js = ', CAGR_gs, CAGR_js)

        # 사용자가 입력한 월(月)만큼 while문 반복 수행
        i = 0
        while i < remain_count:
            gs = gs * (1+CAGR_gs)
            js = js * (1+CAGR_js)

            fill_price_lists(i, gs, js, lr, MAX_PRICE, MIN_PRICE, temp_price_list, price_list)

            i += 1

        # sum구하기 
        sum_price = calc_sum_price(temp_prev_price_list, temp_price_list)

        # prev+pred 가격 리스트
        all_price_list = prev_price_list + price_list
        print('all_price_list = ', all_price_list)

        return prev_count, date_list, sum_price, all_price_list


    # gs_percent, js_percent를 직접 입력했을때
    elif CONDITION3:

        CAGR_gs, CAGR_js = CAGR(gs, js, gs_percent, js_percent, remain_count)
        print('CAGR_gs, CAGR_js = ', CAGR_gs, CAGR_js)

        # 사용자가 입력한 월(月)만큼 while문 반복 수행
        i = 0
        while i < remain_count:
            gs = gs * (1+CAGR_gs)
            js = js * (1+CAGR_js)

            fill_price_lists(i, gs, js, lr, MAX_PRICE, MIN_PRICE, temp_price_list, price_list)

            i += 1

        # sum구하기 
        sum_price = calc_sum_price(temp_prev_price_list, temp_price_list)

        # prev+pred 가격 리스트
        all_price_list = prev_price_list + price_list
        print('all_price_list = ', all_price_list)

        return prev_count, date_list, sum_price, all_price_list

    else:
        print('CONDITION 에러 발생! : 셋 다 입력하면 안돼요.')
        return


def R2(option):
    """[대외변수를 이용해 구해놓은 R^2 값을 가져옵니다.]

    Args:
        option ([int]]): [계정 코드]

    Returns:
        [type]: [r^2 값]
    """
    if option == '100':     # 전사 매출(100)
        return 0.3389       # gs, js 둘다 학습에 적용 했다 가정했을때
    elif option == '300':   # 국내 매출(300)
        return 0.2632       # gs, js 둘다 학습에 적용 했다 가정했을때
