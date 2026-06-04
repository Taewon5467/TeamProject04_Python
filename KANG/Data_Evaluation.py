import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append('KANG')
import CLEAN

df = CLEAN.df_all.copy()

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

print("\n[그래프 1] 시간대별 수요 추이를 화면에 출력합니다...")
hourly_avg = df.groupby('시간', observed=False)['주문건수'].mean().reset_index()

fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.lineplot(data=hourly_avg, x='시간', y='주문건수', marker='o', color='#1f77b4', linewidth=2.5, ax=ax1)
ax1.set_title('안산시 상록구 시간대별 평균 배달 총수요 추이 (3개년 통합)', fontsize=14, pad=15, fontweight='bold')
ax1.set_xlabel('시간대 (Hour)', fontsize=11)
ax1.set_ylabel('평균 배달 호출 건수', fontsize=11)
ax1.set_xticks(range(0, 24))
ax1.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("[그래프 2] 요일별 평균 배달 가중치 비교 그래프를 화면에 출력합니다...")
df['요일_str'] = df['요일'].astype(str)
weekday_avg = df.groupby('요일_str', observed=False)['주문건수'].mean().reset_index()
weekday_avg_sorted = weekday_avg.sort_values(by='주문건수', ascending=False)

fig2, ax2 = plt.subplots(figsize=(9, 5))
sns.barplot(
    data=weekday_avg_sorted, 
    x='요일_str', 
    y='주문건수', 
    hue='요일_str', 
    palette='YlOrRd_r', 
    ax=ax2, 
    legend=False
)
ax2.set_title('안산시 상록구 요일별 평균 배달 수요 비교 (3개년 통합/정렬 완료)', fontsize=14, pad=15, fontweight='bold')
ax2.set_xlabel('요일 (Day of Week)', fontsize=11)
ax2.set_ylabel('평균 배달 호출 건수', fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for p in ax2.patches:
    ax2.annotate(f"{p.get_height():.1f}건", (p.get_x() + p.get_width() / 2., p.get_height() + 0.1),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()

print("[그래프 3] 기상 악화 시 실제 배달 완료 수치 변동 그래프를 화면에 출력합니다...")
all_dates = df['날짜'].unique()
all_hours = list(range(24))
full_index = pd.MultiIndex.from_product([all_dates, all_hours], names=['날짜', '시간']).to_frame().reset_index(drop=True)

df['시간_int'] = df['시간'].astype(int)
df_grid = pd.merge(full_index, df, left_on=['날짜', '시간'], right_on=['날짜', '시간_int'], how='left')

df_grid['주문건수'] = df_grid['주문건수'].fillna(0.0)
df_grid['강수량'] = df_grid['강수량'].fillna(0.0)
df_grid['적설'] = df_grid['적설'].fillna(0.0)

df_grid['기상상황'] = np.where((df_grid['강수량'] > 0) | (df_grid['적설'] > 0), '우천/강설 시', '맑은 날 (기본)')
weather_avg = df_grid.groupby('기상상황', observed=False)['주문건수'].mean().reset_index()

fig3, ax3 = plt.subplots(figsize=(6, 5))
sns.barplot(
    data=weather_avg, 
    x='기상상황', 
    y='주문건수', 
    hue='기상상황', 
    palette='coolwarm', 
    ax=ax3, 
    legend=False
)
ax3.set_title('기상 변화에 따른 평균 배달 완료 건수 변동 (3개년 통합/공백 보정)', fontsize=14, pad=15, fontweight='bold')
ax3.set_xlabel('기상 상태', fontsize=11)
ax3.set_ylabel('평균 배달 호출 건수', fontsize=11)
ax3.grid(axis='y', linestyle='--', alpha=0.5)

for p in ax3.patches:
    ax3.annotate(f"{p.get_height():.1f}건", (p.get_x() + p.get_width() / 2., p.get_height() / 2.),
                ha='center', va='center', color='white', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n🎉 모든 통합 데이터 평가용 그래프 확인이 완료되었습니다!")