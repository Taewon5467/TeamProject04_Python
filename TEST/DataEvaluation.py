# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# file_path = 'DataSet/ansan_delivery_ml_dataset.csv'
# df = pd.read_csv(file_path)

# print("\n==================================================")
# print("      1. 데이터 무결성 및 기본 구조 평가       ")
# print(f"• 전체 수집 시계열 데이터 개수 (Sample Size): {len(df):,}행")
# print(f"• 수집된 피처(컬럼) 목록: {list(df.columns)}")

# # 결측치 비율 평가
# missing_values = df.isnull().sum()
# print("\n[결측치 검증]")
# for col in df.columns:
#     missing_cnt = missing_values[col]
#     missing_ratio = (missing_cnt / len(df)) * 100
#     print(f" - {col}: {missing_cnt}건 누락 (누락률: {missing_ratio:.2f}%)")

# # 데이터 타입 검증
# print("\n[데이터 타입 검증]")
# print(df.dtypes)

# print("\n==================================================")
# print("      2.주요 변수 통계적 분포 평가     ")
# # 로봇 관제에 결정적인 '주문건수', '기온', '강수량'의 기술통계량 평가
# desc = df[['주문건수', '기온', '강수량', '적설']].describe()
# print(desc)

# # 배달 수요(Target)의 이상치(Outlier) 평가
# q1 = df['주문건수'].quantile(0.25)
# q3 = df['주문건수'].quantile(0.75)
# iqr = q3 - q1
# outlier_boundary = q3 + (1.5 * iqr)
# outliers = df[df['주문건수'] > outlier_boundary]
# print(f"\n• 주문건수 사분위 범위 (IQR): {iqr}")
# print(f"• 통계적 폭증(이상치) 기준 주문량: {outlier_boundary:.1f}건 이상")
# print(f"• 기후/이벤트로 인한 주문 폭증 일시 개수: {len(outliers)}건 (전체의 {len(outliers)/len(df)*100:.2f}%)")


# print("\n==================================================")
# print("      3. 배달 영향 요인별 상관관계 평가       ")
# # 요일별 평균 주문건수 평가 (한글 요일 정렬을 위해 순서 지정)
# weekday_order = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
# df['요일'] = pd.Categorical(df['요일'], categories=weekday_order, ordered=True)
# weekday_summary = df.groupby('요일', observed=False)['주문건수'].mean()

# print("[요일별 평균 배달 총수요 현황]")
# for day, val in weekday_summary.items():
#     print(f" - {day}: 평균 {val:.2f}건 호출")

# # 시간대별(피크타임) 상위 3개 골든타임 평가
# hourly_summary = df.groupby('시간')['주문건수'].mean().reset_index()
# top_hours = hourly_summary.sort_values(by='주문건수', ascending=False).head(3)
# print("\n[상록구 전체 배달 피크타임 TOP 3]")
# for idx, row in top_hours.iterrows():
#     print(f" - {int(row['시간'])}시: 평균 {row['주문건수']:.2f}건 발생")

# # 기상 상황(비)에 따른 수요 변동 평가
# rainy_days = df[df['강수량'] > 0]['주문건수'].mean()
# clear_days = df[df['강수량'] == 0]['주문건수'].mean()
# delivery_increase_ratio = ((rainy_days - clear_days) / clear_days) * 100

# print("\n[기상 요인(우천) 지표 평가]")
# print(f" - 맑은 날 평균 주문량: {clear_days:.2f}건")
# print(f" - 우천 시 평균 주문량: {rainy_days:.2f}건")
# print(f" -> 우천 시 배달 주문량 변동률: {delivery_increase_ratio:+.2f}%")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 전처리가 완료된 통합 데이터셋 불러오기
# (통합 데이터셋 파일이 파이썬 파일과 같은 위치에 있다고 가정합니다)
file_path = 'DataSet/ansan_delivery_ml_dataset.csv'

print(f"'{file_path}' 데이터를 읽어와 품질 평가 시각화를 시작합니다...")
df = pd.read_csv(file_path)

# 2. 그래프 한글 폰트 및 마이너스 기호 깨짐 방지 설정
# 윈도우 환경의 기본 한글 폰트인 '맑은 고딕'을 지정합니다.
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)


# ==========================================================
# [그래프 1] : 시간대별 평균 배달 주문건수 추이 (선 그래프)
# ==========================================================
print("\n[그래프 1] 시간대별 수요 추이를 화면에 출력합니다...")
hourly_avg = df.groupby('시간')['주문건수'].mean().reset_index()

fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.lineplot(data=hourly_avg, x='시간', y='주문건수', marker='o', color='#1f77b4', linewidth=2.5, ax=ax1)
ax1.set_title('안산시 상록구 시간대별 평균 배달 총수요 추이', fontsize=14, pad=15, fontweight='bold')
ax1.set_xlabel('시간대 (Hour)', fontsize=11)
ax1.set_ylabel('평균 배달 호출 건수', fontsize=11)
ax1.set_xticks(range(0, 24))
ax1.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show() # 첫 번째 그래프 창을 띄웁니다. 창을 닫아야 다음 그래프로 넘어갑니다.


# ==========================================================
# [그래프 2] : 요일별 평균 배달 주문건수 비교 (내림차순 정렬 바 그래프)
# ==========================================================
print("[그래프 2] 요일별 평균 배달 가중치 비교 그래프를 화면에 출력합니다...")
weekday_avg = df.groupby('요일')['주문건수'].mean().reset_index()
# 가중치(주문량)가 높은 요일부터 순서대로 정렬
weekday_avg_sorted = weekday_avg.sort_values(by='주문건수', ascending=False)

fig2, ax2 = plt.subplots(figsize=(9, 5))
sns.barplot(data=weekday_avg_sorted, x='요일', y='주문건수', palette='YlOrRd_r', ax=ax2)
ax2.set_title('안산시 상록구 요일별 평균 배달 수요 비교 (정렬됨)', fontsize=14, pad=15, fontweight='bold')
ax2.set_xlabel('요일 (Day of Week)', fontsize=11)
ax2.set_ylabel('평균 배달 호출 건수', fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# 바 차트 상단에 수치 데이터 텍스트 표시
for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.1f}건", (p.get_x() + p.get_width() / 2., p.get_height() + 1),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show() # 두 번째 그래프 창을 띄웁니다.


# ==========================================================
# [그래프 3] : 기상 상황별(우천 여부) 배달 총수요 변동 비교 (바 그래프)
# ==========================================================
print("[그래프 3] 기상 악화 시 실제 배달 완료 수치 변동 그래프를 화면에 출력합니다...")
# 강수량이 0보다 크면 우천 시, 0이면 맑은 날로 범주화
df['기상상황'] = np.where(df['강수량'] > 0, '우천 시 (비/눈)', '맑은 날 (기본)')
weather_avg = df.groupby('기상상황')['주문건수'].mean().reset_index()

fig3, ax3 = plt.subplots(figsize=(6, 5))
sns.barplot(data=weather_avg, x='기상상황', y='주문건수', palette='coolwarm', ax=ax3)
ax3.set_title('기상 변화에 따른 평균 배달 완료 건수 변동', fontsize=14, pad=15, fontweight='bold')
ax3.set_xlabel('기상 상태', fontsize=11)
ax3.set_ylabel('평균 배달 호출 건수', fontsize=11)
ax3.grid(axis='y', linestyle='--', alpha=0.5)

for p in ax3.patches:
    ax3.annotate(f"{p.get_height():.1f}건", (p.get_x() + p.get_width() / 2., p.get_height() / 2.),
                ha='center', va='center', color='white', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show() # 세 번째 그래프 창을 띄웁니다.

print("\n🎉 모든 데이터 평가용 그래프 확인이 완료되었습니다!")