import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

file_map = {
    2022: "GG_SGG_FLOWPOP_202203_4127_안산시.csv",
    2023: "GG_SGG_FLOWPOP_202303_4127_안산시.csv",
    2024: "GG_SGG_FLOWPOP_202403_4127_안산시.csv",
    2025: "T21_GG_SGG_FLOWPOP_202503_안산시.csv"
}

def load_and_clean(year, path):
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    df['CTY_NM'] = df['CTY_NM'].str.strip()
    target_df = df[df['CTY_NM'] == '안산시 상록구'].copy()
    target_df['Year'] = year
    return target_df

# 1. 과거 데이터 통합 및 총합 계산
all_data = []
for year, path in file_map.items():
    all_data.append(load_and_clean(year, path))

all_years_df = pd.concat(all_data)
cnt_cols = [col for col in all_years_df.columns if 'CNT' in col]

# 2. 머신러닝으로 2026년 전체 성장률(Ratio) 구하기
years = np.array(list(file_map.keys())).reshape(-1, 1)
totals = []
for y in file_map.keys():
    # 연도별 전체 유동인구 합산
    totals.append(all_years_df[all_years_df['Year'] == y][cnt_cols].sum().sum())

y_totals = np.array(totals)
model = LinearRegression().fit(years, y_totals)
pred_2026_total = model.predict([[2026]])[0]

# 2025년 대비 2026년의 성장 비중(가중치) 계산
growth_ratio = pred_2026_total / totals[-1] 

# 3. 2025년 데이터를 템플릿으로 사용하여 2026년 생성
# (2025년의 시간대별/날짜별 구체적인 유동 패턴을 그대로 복사)
template_2026 = all_years_df[all_years_df['Year'] == 2025].copy()

# 모든 CNT 컬럼에 성장 비율을 곱해서 2026년 규모로 키움
for col in cnt_cols:
    template_2026[col] = (template_2026[col] * growth_ratio).round(2)

# 4. 날짜 정보 업데이트 (2025 -> 2026)
template_2026['ETL_YMD'] = template_2026['ETL_YMD'].astype(str).str.replace('2025', '2026')

# 최종 저장
template_2026.drop(columns=['Year']).to_csv("GG_SGG_FLOWPOP_202603_상록구_최종예측.csv", index=False, encoding='utf-8-sig')

print(f"예측 완료! 2025년 대비 약 {((growth_ratio-1)*100):.2f}% 성장 반영됨.")