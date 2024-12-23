import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import os

# FutureWarning 무시 설정
warnings.simplefilter(action='ignore', category=FutureWarning)

# 파일 경로
google_file_path = 'new_com.xlsx'
#naver_file_path = 'naver.xlsx'
base_dir = os.path.dirname(google_file_path)  # 파일이 있는 디렉토리 경로

# Excel 파일 읽기
google_data = pd.read_excel(google_file_path)
#naver_data = pd.read_excel(naver_file_path)

# 데이터의 첫 몇 행을 확인하여 데이터 구조를 파악
print(google_data.head())
#print(naver_data.head())

# 모든 컬럼 이름 확인
print(google_data.columns)
#print(naver_data.columns)

columns_to_predict = [
    "쌀", "땅콩", "밀가루", "국수", "라면", "당면", "두부", "시리얼", "케이크", "빵", "떡", "파스타면",
    "국산쇠고기", "수입쇠고기", "돼지고기", "닭고기", "소시지", "햄", "갈치", "명태", "조기", "고등어", "오징어",
    "게", "굴", "조개", "전복", "새우", "마른멸치", "마른오징어", "낙지", "오징어채", "북어채", "어묵", "맛살",
    "참치캔", "젓갈", "우유", "치즈", "발효유", "달걀", "참기름", "식용유", "배추", "상추", "시금치", "양배추",
    "당근", "감자", "고구마", "도라지", "콩나물", "버섯", "오이", "풋고추", "호박", "토마토", "파", "양파",
    "마늘", "브로콜리", "단무지", "김", "맛김", "미역", "초콜릿", "사탕", "껌", "아이스크림", "비스킷", "과자",
    "파이", "설탕", "잼", "꿀", "물엿", "고추가루", "참깨", "생강", "소금", "간장", "된장", "양념소스", "고추장",
    "카레", "식초", "혼합조미료", "스프", "이유식", "김치", "밑반찬", "냉동식품", "즉석식품", "편의점도시락", "삼각김밥",
    "커피", "차", "주스", "두유", "생수", "기능성음료", "탄산음료", "소주", "과실주", "맥주", "막걸리", "약주",
    "양주", "담배", "남자외의", "남자상의", "남자하의", "남자내의", "점퍼", "티셔츠", "스웨터", "청바지", "운동복",
    "양말", "모자", "장갑", "의복수선료", "세탁료", "구두", "운동화", "실내화", "전세", "월세", "설비수리비", "상수도료",
    "하수도료", "공동주택관리비", "쓰레기봉투료", "전기료", "도시가스", "지역난방비", "부탄가스", "장롱", "침대", "거실장",
    "소파", "책상", "의자", "식탁", "싱크대", "침구", "커튼", "전기밥솥", "가스레인지", "전자레인지", "전기레인지",
    "냉장고", "에어컨", "선풍기", "공기청정기", "세탁기", "의료건조기", "식기세척기", "청소기", "보온매트", "식기", "컵",
    "솥", "프라이팬", "냄비", "수저", "밀폐용기", "부엌용용구", "보일러", "건전지", "소형가사용품", "세탁세제",
    "섬유유연제", "전구", "부엌용세제", "청소용세제", "살충제", "방향제", "감기약", "소염진통제", "소화제", "조제약",
    "비타민제", "건강기능식품", "유산균", "병원약품", "마스크", "반창고", "안경", "콘택트렌즈", "외래진료비", "건강검진비",
    "약국조제료", "치과진료비", "치과보철료", "입원진료비", "병원검사료", "자전거", "승용차임차료", "열차료", "도시철도료",
    "시내버스료", "시외버스료", "택시료", "이삿짐운송료", "택배이용료", "우편료", "휴대전화기", "유선전화료", "휴대전화료",
    "인터넷이용료", "TV", "영상음향기기", "컴퓨터", "컴퓨터소모품", "저장장치", "서적", "필기구", "유치원납입금",
    "등록금", "김치찌개백반", "된장찌개백반", "비빔밥", "설렁탕", "갈비탕", "삼계탕", "해물찜", "해장국",
    "불고기", "쇠고기", "돼지갈비", "삼겹살", "오리고기", "냉면", "칼국수", "죽", "초밥", 
    "생선회", "짜장면", "짬뽕", "탕수육", "볶음밥", "돈가스", "스테이크", "스파게티", "라면", 
    "김밥", "떡볶이", "치킨", "햄버거", "피자", "쌀국수", "커피", "소주", "맥주", "막걸리", 
    "구내식당식사비", "도시락", "학교기숙사비", "면도기", "헤어드라이어", "칫솔", "치약", "비누", 
    "샴푸", "바디워시", "화장지", "모발염색약", "구강세정제", "손목시계", "장신구", "가방", 
    "핸드백", "우산", "보험서비스료", "금융수수료", "행정수수료", "시험응시료"
]



def process_and_forecast(data, source_name):
    # 폴더 생성
    output_dir = os.path.join(base_dir, source_name)
    os.makedirs(output_dir, exist_ok=True)
    
    results = []

    for column in columns_to_predict:
        if column in data.columns:
            # Prophet 형식에 맞게 변환
            data_column = data.rename(columns={'날짜': 'ds', column: 'y'})[['ds', 'y']]
            
            # 데이터 형식 변환
            data_column['ds'] = pd.to_datetime(data_column['ds'])

            # 학습 데이터와 예측 구간 설정
            train_data = data_column[data_column['ds'] <= '2023-12-03']
            actual_data = data_column[(data_column['ds'] > '2023-12-03') & (data_column['ds'] <= '2024-05-05')]

            # Prophet 모델 생성 및 학습
            model = Prophet()
            model.fit(train_data)

            # 2024년 5월 5일까지 예측
            future = model.make_future_dataframe(periods=len(actual_data), freq='W')
            forecast = model.predict(future)

            # 예측 결과 저장
            forecasted_values = forecast[['ds', 'yhat']].tail(len(actual_data))
            forecasted_values = forecasted_values.set_index('ds')
            actual_values = actual_data.set_index('ds')

            # 실제값과 예측값의 비율 계산
            ratio = actual_values['y'] / forecasted_values['yhat']
            ratio.name = f'{column}_ratio'
            result = pd.concat([actual_values['y'], forecasted_values['yhat'], ratio], axis=1)
            result.columns = ['actual', 'predicted', 'ratio']
            results.append(result)

            '''
            # 예측 결과와 실제 데이터 비교 그래프
            plt.figure(figsize=(10, 6))
            plt.plot(data_column['ds'], data_column['y'], label='Actual Data', color='blue')
            plt.plot(forecast['ds'], forecast['yhat'], label='Predicted Data', color='red')
            plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='pink', alpha=0.3)
            plt.axvline(x=pd.to_datetime('2023-12-03'), color='gray', linestyle='--')
            plt.legend()
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.title(f'Actual vs Predicted Data for {column} ({source_name})')
            plt.savefig(os.path.join(output_dir, f'{column}_comparison.png'))
            plt.close()
            '''

            # 트렌드 및 구성 요소 시각화            
            fig2 = model.plot_components(forecast)
            # 트렌드 및 구성 요소 시각화
            for ax in fig2.axes:
                ax.axis('off')
                ax.tick_params(axis='both', which='both', length=0)  # Hide ticks
                
            for ax in fig2.axes:
                for line in ax.lines:
                    line.set_linewidth(4)  # Set line width to 2
                    line.set_color('black')
            fig2.savefig(os.path.join(output_dir, f'{column}_components.png'))
            plt.close(fig2)
    
    # 예측 결과 저장
    if results:
        combined_results = pd.concat(results, axis=1)
        combined_results.to_csv(os.path.join(output_dir, f'{source_name}_forecast_results.csv'))

# Process and forecast for both datasets
process_and_forecast(google_data, 'google')
#process_and_forecast(naver_data, 'naver')

