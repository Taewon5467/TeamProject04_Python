import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
import os

file_path = 'DataSet/ansan_delivery_ml_dataset.csv'
if not os.path.exists(file_path):
    file_path = 'ansan_delivery_ml_dataset.csv'
df = pd.read_csv(file_path)

vis_path = 'DataSet/OBS_ASOS_TIM_20260528085209.csv'
if not os.path.exists(vis_path):
    vis_path = 'OBS_ASOS_TIM_20260528085209.csv'

try:
    df_vis = pd.read_csv(vis_path, encoding='utf-8')
except Exception:
    df_vis = pd.read_csv(vis_path, encoding='cp949')

df_vis['일시'] = pd.to_datetime(df_vis['일시'])
df_vis['날짜'] = df_vis['일시'].dt.strftime('%Y-%m-%d')
df_vis['시간'] = df_vis['일시'].dt.hour
df_vis['시정'] = df_vis['시정(10m)']
df_vis_clean = df_vis[['날짜', '시간', '시정']]

df_merged = pd.merge(df, df_vis_clean, on=['날짜', '시간'], how='inner')

holidays_2021 = [
    '2021-01-01', 
    '2021-02-10', '2021-02-11', '2021-02-12', '2021-02-13', 
    '2021-03-01', 
    '2021-05-05', '2021-05-19', 
    '2021-06-06'
]

df_merged['Is_Holiday'] = df_merged['날짜'].isin(holidays_2021).astype(int)
df_merged['Is_PeakTime'] = df_merged['시간'].apply(lambda x: 1 if (11 <= x <= 13) or (17 <= x <= 20) else 0)
df_merged['Outdoor_Activity_Index'] = df_merged['기온'] + (df_merged['강수량'] * 2.5) + (df_merged['적설'] * 4.0)

df_ml = pd.get_dummies(df_merged, columns=['요일'], drop_first=False)

exclude_cols = ['날짜', '시도', '시군구', '업종', '주문건수']
feature_cols = [col for col in df_ml.columns if col not in exclude_cols]

X = df_ml[feature_cols]
y = df_ml['주문건수']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_before = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model_before.fit(X_train, y_train)

train_preds_before = model_before.predict(X_train)
test_preds_before = model_before.predict(X_test)

r2_train_before = r2_score(y_train, train_preds_before)
r2_test_before = r2_score(y_test, test_preds_before)

rmsle_train_before = np.sqrt(mean_squared_log_error(y_train, np.clip(train_preds_before, 0, None)))
rmsle_test_before = np.sqrt(mean_squared_log_error(y_test, np.clip(test_preds_before, 0, None)))

param_grid = {
    'n_estimators': [150, 250, 350],
    'learning_rate': [0.03, 0.05],
    'max_depth': [3, 4],
    'subsample': [0.7, 0.8],
    'min_samples_leaf': [5, 9]
}

grid_search = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

train_preds_after = best_model.predict(X_train)
test_preds_after = best_model.predict(X_test)

r2_train_after = r2_score(y_train, train_preds_after)
r2_test_after = r2_score(y_test, test_preds_after)

rmsle_train_after = np.sqrt(mean_squared_log_error(y_train, np.clip(train_preds_after, 0, None)))
rmsle_test_after = np.sqrt(mean_squared_log_error(y_test, np.clip(test_preds_after, 0, None)))

results = [
    {"Stage": "Before Tuning", "Dataset": "Train", "R2_Score": r2_train_before, "RMSLE": rmsle_train_before},
    {"Stage": "Before Tuning", "Dataset": "Test", "R2_Score": r2_test_before, "RMSLE": rmsle_test_before},
    {"Stage": "After Tuning", "Dataset": "Train", "R2_Score": r2_train_after, "RMSLE": rmsle_train_after},
    {"Stage": "After Tuning", "Dataset": "Test", "R2_Score": r2_test_after, "RMSLE": rmsle_test_after}
]
comparison_df = pd.DataFrame(results)

print("\n" + "="*65)
print("       [최종 결과] 훈련(Train) vs 테스트(Test) 종합 지표")
print("="*65)
print(comparison_df.to_string(index=False))
print("="*65 + "\n")

plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

fig1, ax1 = plt.subplots(figsize=(7.5, 5))
sns.barplot(data=comparison_df, x='Stage', y='R2_Score', hue='Dataset', palette='Blues', ax=ax1, width=0.5)
ax1.set_title('하이퍼파라미터 튜닝 전/후 Train vs Test $R^2$ Score 비교', fontsize=11, fontweight='bold', pad=12)
ax1.set_ylabel('$R^2$ Score', fontsize=10)
ax1.set_xlabel('튜닝 단계', fontsize=10)
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

for container in ax1.containers:
    ax1.bar_label(container, fmt='%.4f', padding=3, fontweight='bold', fontsize=9, color='#2c3e50')

plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(7.5, 5))
sns.barplot(data=comparison_df, x='Stage', y='RMSLE', hue='Dataset', palette='Oranges', ax=ax2, width=0.5)
ax2.set_title('하이퍼파라미터 튜닝 전/후 Train vs Test RMSLE 비교', fontsize=11, fontweight='bold', pad=12)
ax2.set_ylabel('RMSLE Score (Error)', fontsize=10)
ax2.set_xlabel('튜닝 단계', fontsize=10)
ax2.set_ylim(0, max(comparison_df['RMSLE']) * 1.2)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for container in ax2.containers:
    ax2.bar_label(container, fmt='%.4f', padding=3, fontweight='bold', fontsize=9, color='#2c3e50')

plt.tight_layout()
plt.show()

fig3, ax3 = plt.subplots(figsize=(14, 5.5)) 
sample_size = 60
x_axis = np.arange(sample_size)

ax3.plot(x_axis, y_test.values[:sample_size], label='Actual (실제 주문량)', color='black', linewidth=2.5, linestyle='--')
ax3.plot(x_axis, test_preds_before[:sample_size], label='Before Tuning (Test Pred)', color='#7fcdbb', linewidth=1.5, alpha=0.8)
ax3.plot(x_axis, test_preds_after[:sample_size], label='After Tuning (Test Pred Best)', color='#253494', linewidth=2.5)

ax3.set_title('실제 배달량 vs 시정 반영 모델 튜닝 전/후 예측 흐름 대조', fontsize=12, fontweight='bold', pad=15)
ax3.set_xlabel('테스트 데이터 샘플 인덱스 (시간 흐름)', fontsize=10)
ax3.set_ylabel('배달 주문량 (건)', fontsize=10)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(bbox_to_anchor=(1.01, 0.95), loc='upper left', borderaxespad=0, fontsize=10)

plt.tight_layout()
plt.show()