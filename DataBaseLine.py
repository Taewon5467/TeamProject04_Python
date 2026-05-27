import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error

file_path = 'DataSet/ansan_delivery_ml_dataset.csv'
print(f"[{file_path}] 데이터를 로드...")
df = pd.read_csv(file_path)

df_ml = pd.get_dummies(df, columns=['요일'], drop_first=False)

exclude_cols = ['날짜', '시도', '시군구', '업종', '주문건수']
feature_cols = [col for col in df_ml.columns if col not in exclude_cols]

X = df_ml[feature_cols]
y = df_ml['주문건수']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

baseline_models = {
    "단순 평균 (Mean Dummy)": DummyRegressor(strategy="mean"),
    "중앙값 (Median Dummy)": DummyRegressor(strategy="median"),
    "선형 회귀 (Linear Regression)": LinearRegression()
}

results_list = []
pred_dict = {}

for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    
    preds_clipped = np.clip(preds, 0, None)
    rmsle = np.sqrt(mean_squared_log_error(y_test, preds_clipped))
    
    results_list.append({"Model": name, "R2_Score": r2, "RMSLE": rmsle})
    pred_dict[name] = preds

comparison_df = pd.DataFrame(results_list).sort_values(by="R2_Score", ascending=False).reset_index(drop=True)

print("\n" + "="*55)
print("       [종합 결과] 베이스라인 3종 R2 및 RMSLE 비교 지표")
print("="*55)
print(comparison_df.to_string(index=False))
print("="*55 + "\n")

print("그래프 시각화...")
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

print(" -> [창 1] R2 Score 비교 막대그래프")
fig1, ax1 = plt.subplots(figsize=(6.5, 5))

sns.barplot(data=comparison_df, x='Model', y='R2_Score', palette='pastel', ax=ax1, width=0.4)
ax1.set_title('기본 베이스라인 3종 예측력 ($R^2$ Score) 비교\n(1에 가까울수록 우수)', fontsize=11, fontweight='bold', pad=12)
ax1.set_ylabel('$R^2$ Score', fontsize=10)
ax1.set_xlabel('베이스라인 모델 종류', fontsize=10)
ax1.set_ylim(-0.05, 0.5)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

for p in ax1.patches:
    ax1.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                ha='center', va='center', fontsize=10, fontweight='bold', color='#2c3e50')

plt.tight_layout()
plt.show()

print(" -> [창 2] RMSLE 오차 지표 비교 막대그래프")
fig3, ax3 = plt.subplots(figsize=(6.5, 5))

comparison_df_rmsle = comparison_df.sort_values(by="RMSLE", ascending=True).reset_index(drop=True)

sns.barplot(data=comparison_df_rmsle, x='Model', y='RMSLE', palette='autumn_r', ax=ax3, width=0.4)
ax3.set_title('기본 베이스라인 3종 오차 지표 (RMSLE) 비교\n(0에 가까울수록 오차가 적고 우수)', fontsize=11, fontweight='bold', pad=12)
ax3.set_ylabel('RMSLE Score (Error)', fontsize=10)
ax3.set_xlabel('베이스라인 모델 종류', fontsize=10)
ax3.set_ylim(0, max(comparison_df_rmsle['RMSLE']) * 1.2)
ax3.grid(axis='y', linestyle='--', alpha=0.5)

for p in ax3.patches:
    ax3.annotate(f"{p.get_height():.4f}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.01),
                ha='center', va='center', fontsize=10, fontweight='bold', color='#2c3e50')

plt.tight_layout()
plt.show()

print(" -> [창 3] 시계열 추이 비교 그래프")
fig2, ax2 = plt.subplots(figsize=(14, 5)) 

sample_size = 60
x_axis = np.arange(sample_size)

ax2.plot(x_axis, y_test.values[:sample_size], label='Actual (실제 주문량)', color='black', linewidth=2.5, linestyle='--')
ax2.plot(x_axis, pred_dict["단순 평균 (Mean Dummy)"][:sample_size], label='단순 평균 (Mean)', color='#ff9800', linewidth=1.5, linestyle=':')
ax2.plot(x_axis, pred_dict["중앙값 (Median Dummy)"][:sample_size], label='중앙값 (Median)', color='#e91e63', linewidth=1.5, linestyle='-.')
ax2.plot(x_axis, pred_dict["선형 회귀 (Linear Regression)"][:sample_size], label='선형 회귀 (Linear)', color='#2196f3', linewidth=2.0)

ax2.set_title(f'실제 배달량 vs 베이스라인 3종 예측 흐름 비교 (샘플 {sample_size}개 구간 관찰)', fontsize=12, fontweight='bold', pad=12)
ax2.set_xlabel('테스트 데이터 샘플 인덱스 (시간 흐름)', fontsize=10)
ax2.set_ylabel('배달 주문량 (건)', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()

print("\n두 가지 평가 그래프의 분리 및 확장 출력이 정상 종료")