from flask import Flask, render_template, request
import matplotlib.pyplot as plt

from util import *

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])

def index():
    if request.method == 'GET':
      return render_template('index.html')
    if request.method == 'POST':
        option = str(request.form['options'])
        gs_percent = float(request.form['gs_percent'])
        js_percent = float(request.form['js_percent'])
        all_percent = float(request.form['all_percent'])
        start_date = str(request.form['start_date']) # 2020-07
        end_date = str(request.form['end_date']) # 2020-12

        count = 0 # 총 개월 수
        year_count = int(end_date[:4]) - int(start_date[:4])        # 년 차이
        month_count = int(end_date[5:]) - int(start_date[5:]) + 1   # 월 차이
        count = year_count*12 + month_count
        print('(server.py) count = ', count)

        r2 = R2(option)
        r2 = round(r2, 2) # r2값은 %말고, 소수점으로 표현

        try:
            # 예측 값을 배열로 리턴 
            prev_count, date_list, sum_price, all_price_list \
                = price_predict(gs_percent, js_percent, all_percent, option, start_date, end_date, count)
        except:
            return render_template('500.html')
     
        # code -> 글자로 변경
        if option == '100':
            option = '전사매출'
        elif option == '300':
            option = '국내매출'
        else:
            pass

        # 전월 대비 증감(=all_percent)을 선택했다면..
        if int(all_percent) != 0: 
            return render_template('index.html', all_price_list=all_price_list,
                                        now_selected_percent0=all_percent,
                                        option=option, count=count, date_list=date_list,
                                        r2=r2, sum_price=sum_price, prev_count=prev_count)
        # 전체 증감(%)으로 입력하지 않았을 때..
        else:
            print('여기')
            return render_template('index.html', all_price_list=all_price_list, 
                                        now_selected_percent1=gs_percent, now_selected_percent2=js_percent,
                                        option=option, count=count, date_list=date_list,
                                        r2=r2, sum_price=sum_price, prev_count=prev_count)

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(host = '0.0.0.0') # AWS EC2 호스팅할때, 사용 