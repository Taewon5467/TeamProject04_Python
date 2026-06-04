import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

df2019 = pd.read_csv("DataSet/ansan_delivery_2019_clean.csv")
df2020 = pd.read_csv("DataSet/ansan_delivery_2020_clean.csv")
df2021 = pd.read_csv("DataSet/ansan_delivery_2021_clean.csv")

df2019["연도"] = 2019
df2020["연도"] = 2020
df2021["연도"] = 2021

df = pd.concat([df2019, df2020, df2021], ignore_index=True)

if "업종" in df.columns:
    df = df.drop(columns=["업종"])
    
df["월"] = df["날짜"].str[:2].astype(int)
df["일"] = df["날짜"].str[3:].astype(int)
df["주말"] = df["요일"].isin(["토요일", "일요일"]).astype(int)

def season(month):
    if month in [3, 4, 5]:
        return "봄"
    elif month in [6, 7, 8]:
        return "여름"
    elif month in [9, 10, 11]:
        return "가을"
    return "겨울"

df["계절"] = df["월"].apply(season)

print("\n===== 강수량 통계 =====")
print(df["강수량"].describe())

rain_df = df.copy()

rain_df["비구간"] = pd.cut(
    rain_df["강수량"],
    bins=[-1, 0, 1, 5, 10, 100],
    labels=["0", "0~1", "1~5", "5~10", "10+"]
)

print("\n===== 강수량별 평균 주문건수 =====")
print(
    rain_df.groupby("비구간")["주문건수"]
           .mean()
)

print("\n===== 요일별 평균 주문건수 =====")

print(
    df.groupby("요일")["주문건수"]
      .mean()
      .sort_values(ascending=False)
)

X = df.drop(columns=["주문건수", "날짜"])
y = df["주문건수"]

categorical_features = ["요일", "시도", "시군구", "계절"]
numeric_features = ["시간", "기온", "강수량", "적설", "시정", "연도", "월", "주말"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features)
    ],
    verbose_feature_names_out=False
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1))
])

param_grid = {
    "model__n_estimators": [100, 300, 500],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__num_leaves": [15, 31, 63],
    "model__max_depth": [4, 6, -1]
}

print("🚀 LightGBM 하이퍼파라미터 GridSearchCV 최적화 수행 중...")
grid_search = GridSearchCV(
    base_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\n✅ 최적 파라미터 조합: {grid_search.best_params_}")
print(f"✅ 최적 교차 검증 R² Score: {grid_search.best_score_:.4f}")

best_pipeline = grid_search.best_estimator_

X_test_trans = best_pipeline.named_steps["preprocessor"].transform(X_test)
if hasattr(X_test_trans, "toarray"):
    X_test_trans = X_test_trans.toarray()

pred = best_pipeline.named_steps["model"].predict(X_test_trans)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

performance_df = pd.DataFrame([
    {"Metric": "MAE", "Value": round(mae, 4)},
    {"Metric": "RMSE", "Value": round(rmse, 4)},
    {"Metric": "R² Score", "Value": round(r2, 4)}
])

print("\n" + "="*40)
print(" 튜닝된 LightGBM 최종 테스트 데이터 평가")
print("="*40)
print(performance_df.to_string(index=False))
print("="*40)

feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = best_pipeline.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)
top20 = importance_df.head(20)

print("\n===== 전체 Feature Importance =====")

print(
    importance_df
    .sort_values("Importance", ascending=False)
    .to_string(index=False)
)

# ===========================
# 요일 Feature만 따로 출력
# ===========================
print("\n===== 요일 중요도 =====")

weekday_features = importance_df[
    importance_df["Feature"].str.contains("요일")
]

print(
    weekday_features
    .sort_values("Importance", ascending=False)
    .to_string(index=False)
)

# ===========================
# 그래프 생성
# ===========================
fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.barh(top20["Feature"], top20["Importance"], color="#2c3e50")
ax1.invert_yaxis()
ax1.set_title("LightGBM 최적 모델 피처 중요도 (Top 20)", fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel("Importance")
plt.tight_layout()

fig2, ax2 = plt.subplots(figsize=(8, 8))
ax2.scatter(y_test, pred, alpha=0.5, color="#e74c3c")
ax2.set_xlabel("Actual")
ax2.set_ylabel("Predicted")
ax2.set_title("실제 주문건수 vs 최적화 모델 예측건수 비교", fontsize=14, fontweight='bold', pad=15)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="black", linestyle="--", linewidth=2)
plt.tight_layout()

plt.show()

import joblib

feat_order = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
pipeline_save_data = {
    'model': best_pipeline, 
    'features': feat_order # 학습할 때 사용한 피처 리스트
}
joblib.dump(pipeline_save_data, 'lgbm_delivery_pipeline.pkl')
print("\n✅ 모델 저장 완료: lgbm_delivery_pipeline.pkl")