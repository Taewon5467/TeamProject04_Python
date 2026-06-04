import pandas as pd

# 1. 파일 경로 설정
delivery_file_path = 'DataSet/KGU_3rd_ORIGIN_KGUDSTNORDER_20210101000000.csv'
weather_file_path = 'DataSet/OBS_ASOS_TIM_20260526152623.csv'

print("1. 배달 원본 데이터를 읽어와 안산시 상록구 시간별 데이터로 정제하는 중...")
df_delivery = pd.read_csv(delivery_file_path, header=None)

# 안산시 상록구 필터링 및 시간별 합산
ansan_sangnok_df = df_delivery[df_delivery[4] == '안산시 상록구'].copy()
grouped_delivery = ansan_sangnok_df.groupby([0, 1, 3, 4])[5].sum().reset_index()
grouped_delivery['업종'] = '음식'
final_delivery = grouped_delivery[[0, 1, '업종', 3, 4, 5]]
final_delivery.columns = ['날짜', '시간', '업종', '시도', '시군구', '주문건수']


print("2. 날씨 원본 데이터를 읽어와 정제하는 중...")
df_weather = pd.read_csv(weather_file_path, encoding='cp949')

# 날씨 데이터의 일시를 날짜와 시간으로 분리
df_weather['일시'] = pd.to_datetime(df_weather['일시'])
df_weather['날짜'] = df_weather['일시'].dt.strftime('%Y-%m-%d')
df_weather['시간'] = df_weather['일시'].dt.hour

# 컬럼명 매핑 및 결측치 처리
df_weather['기온'] = df_weather['기온(°C)']
df_weather['강수량'] = df_weather['강수량(mm)'].fillna(0.0)
df_weather['적설'] = df_weather['적설(cm)'].fillna(0.0)
final_weather = df_weather[['날짜', '시간', '기온', '강수량', '적설']]


print("3. [날짜]와 [시간]을 기준으로 배달 데이터와 날씨 데이터 병합 중...")
merged_df = pd.merge(final_delivery, final_weather, on=['날짜', '시간'], how='inner')


print("4. [★업그레이드] 날짜를 기준으로 '요일'을 추출한 뒤 한글로 변환 중...")
# 문자열 날짜를 숫자로 먼저 추출 (0=월, 6=일)
merged_df['날짜'] = pd.to_datetime(merged_df['날짜'])
weekday_numeric = merged_df['날짜'].dt.dayofweek

# 숫자를 한글 요일 이름으로 매핑
weekday_mapping = {
    0: '월요일',
    1: '화요일',
    2: '수요일',
    3: '목요일',
    4: '금요일',
    5: '토요일',
    6: '일요일'
}

merged_df['요일'] = weekday_numeric.map(weekday_mapping)

merged_df['날짜'] = merged_df['날짜'].dt.strftime('%Y-%m-%d')


print("5. 머신러닝 학습이 최적화되도록 컬럼 순서 재배치 중...")
final_dataset = merged_df[['날짜', '시간', '요일', '기온', '강수량', '적설', '시도', '시군구', '업종', '주문건수']]

output_file = 'ansan_delivery_ml_dataset.csv'
final_dataset.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n🎉 한글 요일이 포함된 최종 데이터셋 통합 성공!")
print(f"💾 파일이 '{output_file}'로 성공적으로 저장되었습니다.")
print(f"📊 총 시계열 행 데이터 개수: {len(final_dataset):,}개")

print("\n--- 최종 결합본 데이터 샘플 확인 (상위 5줄) ---")
print(final_dataset[['날짜', '시간', '요일', '기온', '주문건수']].head())