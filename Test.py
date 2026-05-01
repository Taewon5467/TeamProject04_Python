import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. 학습 데이터 (2022~2024) 및 실제 데이터 (2025) 설정
train_files = {
    2022: "GG_SGG_FLOWPOP_202203_4127_안산시.csv",
    2023: "GG_SGG_FLOWPOP_202303_4127_안산시.csv",
    2024: "GG_SGG_FLOWPOP_202403_4127_안산시.csv"
}
actual_2025_file = "T21_GG_SGG_FLOWPOP_202503_안산시.csv"

def get_sangnok_total(filepath):
    df = pd.read_csv(filepath)
    df.columns = [col.strip() for col in df.columns]
    # 상록구 데이터 필터링
    df_sangnok = df[df['CTY_NM'].str.strip() == '안산시 상록구']
    cnt_cols = [col for col in df_sangnok.columns if 'CNT' in col]
    return df_sangnok[cnt_cols].sum().sum()

# 2. 연도별 총량 수집 및 머신러닝 학습
years = sorted(train_files.keys())
totals = [get_sangnok_total(train_files[y]) for y in years]

X = np.array(years).reshape(-1, 1)
y = np.array(totals)

model = LinearRegression().fit(X, y)

# 3. 2025년 예측 수행
pred_2025_total = model.predict([[2025]])[0]

# 4. 실제 2025년 데이터와 비교 검증
actual_2025_total = get_sangnok_total(actual_2025_file)
accuracy = (1 - abs(actual_2025_total - pred_2025_total) / actual_2025_total) * 100

print(f"--- 2025년 예측 성능 테스트 ---")
print(f"2022~2024 기반 2025 예측치: {int(pred_2025_total):,}명")
print(f"2025년 실제 유동인구: {int(actual_2025_total):,}명")
print(f"모델 정확도: {accuracy:.2f}%")

# 5. 테스트용 예측 CSV 생성 (패턴 보존 방식)
# 2024년 데이터를 템플릿으로 사용
df_2024 = pd.read_csv("GG_SGG_FLOWPOP_202403_4127_안산시.csv")
df_2024.columns = [col.strip() for col in df_2024.columns]
template_2025 = df_2024[df_2024['CTY_NM'].str.strip() == '안산시 상록구'].copy()

# 성장 비율 적용 (2025 예측치 / 2024 실제치)
growth_ratio_test = pred_2025_total / totals[-1]
cnt_cols = [col for col in template_2025.columns if 'CNT' in col]

for col in cnt_cols:
    template_2025[col] = (template_2025[col] * growth_ratio_test).round(2)

template_2025.to_csv("TEST_sangnok_2025_prediction.csv", index=False, encoding='utf-8-sig')
print("\n검증용 파일 생성 완료: TEST_sangnok_2025_prediction.csv")