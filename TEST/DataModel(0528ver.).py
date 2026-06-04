import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_log_error
import lightgbm as lgb
import warnings
import os

warnings.filterwarnings('ignore')
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

file_path = 'DataSet/ansan_delivery_ml_dataset.csv' 
if not os.path.exists(file_path):
    file_path = 'ansan_delivery_ml_dataset.csv'
df = pd.read_csv(file_path)

vis_path = 'DataSet/OBS_ASOS_TIM_20260528085209.csv' # 0528 추가. 시정 데이터 파일
if not os.path.exists(vis_path):
    vis_path = 'OBS_ASOS_TIM_20260528085209.csv'

if os.path.exists(vis_path):
    try:
        df_vis = pd.read_csv(vis_path, encoding='utf-8')
    except Exception:
        df_vis = pd.read_csv(vis_path, encoding='cp949')
    df_vis['일시'] = pd.to_datetime(df_vis['일시'])
    df_vis['날짜'] = df_vis['일시'].dt.strftime('%Y-%m-%d')
    df_vis['시간'] = df_vis['일시'].dt.hour
    df_vis['시정'] = df_vis['시정(10m)']
    df = pd.merge(df, df_vis[['날짜','시간','시정']], on=['날짜','시간'], how='inner')
    print("시정 데이터 병합 완료")

holidays_2021 = [ # 2021년 공휴일
    '2021-01-01',
    '2021-02-10','2021-02-11','2021-02-12','2021-02-13',
    '2021-03-01',
    '2021-05-05','2021-05-19',
    '2021-06-06'
]
df['Is_Holiday']  = df['날짜'].isin(holidays_2021).astype(int)
df['Is_PeakTime'] = df['시간'].apply(lambda x: 1 if (11<=x<=13) or (17<=x<=20) else 0) # 점심/저녁 피크타임 여부
df['Outdoor_Activity_Index'] = df['기온'] + (df['강수량']*2.5) + (df['적설']*4.0) # 야외활동지수 (낮을수록 나쁨)

df_before = pd.get_dummies(df.copy(), columns=['요일'], drop_first=False)
excl = ['날짜','시도','시군구','업종','주문건수']
feat_before = [c for c in df_before.columns if c not in excl]

X_before = df_before[feat_before]
y         = df_before['주문건수']

X_tr_b, X_te_b, y_tr, y_te = train_test_split(
    X_before, y, test_size=0.2, random_state=42
) # 동일 random_state로 분리하여 이후 단계에서도 재사용 (y_tr, y_te)
 
# Before Tuning
gbr_raw = GradientBoostingRegressor(
    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
)
gbr_raw.fit(X_tr_b, y_tr)
p_tr_raw = gbr_raw.predict(X_tr_b)
p_te_raw = gbr_raw.predict(X_te_b)

# After Tuning (GridSearchCV)
print("GridSearchCV 실행 중...")
param_grid = {
    'n_estimators'   : [150, 250, 350],
    'learning_rate'  : [0.03, 0.05],
    'max_depth'      : [3, 4],
    'subsample'      : [0.7, 0.8],
    'min_samples_leaf': [5, 9]
}
gs = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=0
)
gs.fit(X_tr_b, y_tr)
gbr_tuned = gs.best_estimator_
p_tr_tuned = gbr_tuned.predict(X_tr_b)
p_te_tuned = gbr_tuned.predict(X_te_b)
print(f"최적 파라미터: {gs.best_params_}")

# AFTER 피처 (개선된 피처 + LightGBM)
df2 = df.copy()
df2['날짜_dt'] = pd.to_datetime(df2['날짜']) # 날짜를 datetime으로 변환하여 월/일 추출
df2['월'] = df2['날짜_dt'].dt.month
df2['일'] = df2['날짜_dt'].dt.day

df2['Is_Weekend']   = df2['요일'].isin(['토요일','일요일']).astype(int) # 주말 여부
df2['Is_Lunch']     = ((df2['시간']>=11)&(df2['시간']<=13)).astype(int) # 점심 피크타임 여부
df2['Is_Dinner']    = ((df2['시간']>=17)&(df2['시간']<=20)).astype(int) # 저녁 피크타임 여부
df2['Is_LateNight'] = ((df2['시간']>=21)|(df2['시간']<=3)).astype(int) # 야간 피크타임 여부

# == AI 코드 ==
df2['시간_sin'] = np.sin(2*np.pi*df2['시간']/24) # 시간의 주기성을 사인/코사인으로 표현
df2['시간_cos'] = np.cos(2*np.pi*df2['시간']/24)
df2['월_sin']   = np.sin(2*np.pi*df2['월']/12)
df2['월_cos']   = np.cos(2*np.pi*df2['월']/12)
day_map = {'월요일':0,'화요일':1,'수요일':2,'목요일':3,'금요일':4,'토요일':5,'일요일':6}
df2['요일_num'] = df2['요일'].map(day_map)
df2['요일_sin'] = np.sin(2*np.pi*df2['요일_num']/7)
df2['요일_cos'] = np.cos(2*np.pi*df2['요일_num']/7)

df2['기온_강수'] = df2['기온'] * df2['강수량']
df2['기온_제곱'] = df2['기온'] ** 2
df2['강수_있음'] = (df2['강수량']>0).astype(int)
df2['적설_있음'] = (df2['적설']>0).astype(int)
df2['쾌적도']   = -np.abs(df2['기온']-17.5)
df2['Outdoor_Activity_Index'] = df2['기온']-(df2['강수량']*2.5)-(df2['적설']*4.0)

df2['log_주문건수'] = np.log1p(df2['주문건수'])
df2 = pd.get_dummies(df2, columns=['요일'], drop_first=False)

excl2 = ['날짜','시도','시군구','업종','주문건수','log_주문건수','날짜_dt']
feat_after = [c for c in df2.columns if c not in excl2]

X_after  = df2[feat_after]
y_log    = df2['log_주문건수']
# ====
# 동일 random_state로 분리
X_tr_a, X_te_a, yl_tr, yl_te = train_test_split(
    X_after, y_log, test_size=0.2, random_state=42
)

print("LightGBM 학습 중...")
lgbm = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    num_leaves=40, min_child_samples=10,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.5,
    random_state=42, verbose=-1
)
lgbm.fit(
    X_tr_a, yl_tr,
    eval_set=[(X_te_a, yl_te)],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)]
)
p_tr_lgbm = np.expm1(lgbm.predict(X_tr_a))
p_te_lgbm = np.expm1(lgbm.predict(X_te_a))

# 지표 계산
def metrics(y_true, y_pred):
    r2    = r2_score(y_true, y_pred)
    rmsle = np.sqrt(mean_squared_log_error(y_true, np.clip(y_pred, 0, None)))
    return round(r2, 4), round(rmsle, 4)

r2_tr_raw,   rmsle_tr_raw   = metrics(y_tr, p_tr_raw)
r2_te_raw,   rmsle_te_raw   = metrics(y_te, p_te_raw)
r2_tr_tuned, rmsle_tr_tuned = metrics(y_tr, p_tr_tuned)
r2_te_tuned, rmsle_te_tuned = metrics(y_te, p_te_tuned)
r2_tr_lgbm,  rmsle_tr_lgbm  = metrics(y_tr, p_tr_lgbm)
r2_te_lgbm,  rmsle_te_lgbm  = metrics(y_te, p_te_lgbm)

results = pd.DataFrame([
    {"단계":"Before Tuning (원본 GBR)", "데이터":"Train", "R² Score":r2_tr_raw,   "RMSLE":rmsle_tr_raw},
    {"단계":"Before Tuning (원본 GBR)", "데이터":"Test",  "R² Score":r2_te_raw,   "RMSLE":rmsle_te_raw},
    {"단계":"After Tuning (원본 GBR)",  "데이터":"Train", "R² Score":r2_tr_tuned, "RMSLE":rmsle_tr_tuned},
    {"단계":"After Tuning (원본 GBR)",  "데이터":"Test",  "R² Score":r2_te_tuned, "RMSLE":rmsle_te_tuned},
    {"단계":"After Tuning (LightGBM)",  "데이터":"Train", "R² Score":r2_tr_lgbm,  "RMSLE":rmsle_tr_lgbm},
    {"단계":"After Tuning (LightGBM)",  "데이터":"Test",  "R² Score":r2_te_lgbm,  "RMSLE":rmsle_te_lgbm},
])

print("\n" + "="*70)
print("   Train / Test × Before / After 종합 비교")
print("="*70)
print(results.to_string(index=False))
print("="*70)

# 시각화 설정 (AI)
BLUE_TR   = '#378ADD'   # Train Before
BLUE_TR2  = '#185FA5'   # Train After (GBR tuned)
BLUE_TR3  = '#042C53'   # Train After (LightGBM)
TEAL_TE   = '#5DCAA5'   # Test Before
TEAL_TE2  = '#0F6E56'   # Test After (GBR tuned)
TEAL_TE3  = '#04342C'   # Test After (LightGBM)

AMB_TR    = '#EF9F27'
AMB_TR2   = '#854F0B'
AMB_TR3   = '#412402'
COR_TE    = '#F0997B'
COR_TE2   = '#993C1D'
COR_TE3   = '#4A1B0C'

stages    = ['Before\nTuning', 'After Tuning\n(GBR)', 'After Tuning\n(LightGBM)']
x         = np.arange(len(stages))
w         = 0.32

# ── Fig 1: R² Score ──────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(9, 5))

tr_r2 = [r2_tr_raw, r2_tr_tuned, r2_tr_lgbm]
te_r2 = [r2_te_raw, r2_te_tuned, r2_te_lgbm]

b_tr = ax1.bar(x - w/2, tr_r2, width=w,
               color=[BLUE_TR, BLUE_TR2, BLUE_TR3],
               label='Train', zorder=3)
b_te = ax1.bar(x + w/2, te_r2, width=w,
               color=[TEAL_TE, TEAL_TE2, TEAL_TE3],
               label='Test',  zorder=3)

ax1.set_xticks(x)
ax1.set_xticklabels(stages, fontsize=10)
ax1.set_ylabel('R² Score', fontsize=10)
ax1.set_ylim(0.65, 1.03)
ax1.set_title('Train vs Test  ×  Before / After — R² Score 비교',
              fontsize=12, fontweight='bold', pad=14)
ax1.grid(axis='y', linestyle='--', alpha=0.45, zorder=0)
ax1.spines[['top','right']].set_visible(False)

for bar in list(b_tr) + list(b_te):
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.004,
             f'{h:.4f}', ha='center', va='bottom',
             fontsize=8.5, fontweight='bold', color='#2c3e50')

# 개선 화살표 (Test: Before → LightGBM)
ax1.annotate('',
    xy=(x[2]+w/2, r2_te_lgbm), xytext=(x[0]+w/2, r2_te_raw),
    arrowprops=dict(arrowstyle='->', color='#0F6E56', lw=1.8,
                    connectionstyle='arc3,rad=-0.25'))
ax1.text(1.55, (r2_te_raw + r2_te_lgbm)/2 + 0.012,
         f'+{r2_te_lgbm-r2_te_raw:.4f}', color='#0F6E56',
         fontsize=9, fontweight='bold')

legend_patches = [
    plt.Rectangle((0,0),1,1, color=BLUE_TR,  label='Train · Before Tuning'),
    plt.Rectangle((0,0),1,1, color=BLUE_TR2, label='Train · After Tuning (GBR)'),
    plt.Rectangle((0,0),1,1, color=BLUE_TR3, label='Train · After Tuning (LightGBM)'),
    plt.Rectangle((0,0),1,1, color=TEAL_TE,  label='Test · Before Tuning'),
    plt.Rectangle((0,0),1,1, color=TEAL_TE2, label='Test · After Tuning (GBR)'),
    plt.Rectangle((0,0),1,1, color=TEAL_TE3, label='Test · After Tuning (LightGBM)'),
]
ax1.legend(handles=legend_patches, fontsize=8, loc='lower right',
           ncol=2, framealpha=0.9)

plt.tight_layout()
plt.show()

# ── Fig 2: RMSLE ─────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5))

tr_rmsle = [rmsle_tr_raw, rmsle_tr_tuned, rmsle_tr_lgbm]
te_rmsle = [rmsle_te_raw, rmsle_te_tuned, rmsle_te_lgbm]

b_tr2 = ax2.bar(x - w/2, tr_rmsle, width=w,
                color=[AMB_TR, AMB_TR2, AMB_TR3],
                label='Train', zorder=3)
b_te2 = ax2.bar(x + w/2, te_rmsle, width=w,
                color=[COR_TE, COR_TE2, COR_TE3],
                label='Test',  zorder=3)

ax2.set_xticks(x)
ax2.set_xticklabels(stages, fontsize=10)
ax2.set_ylabel('RMSLE  (낮을수록 좋음)', fontsize=10)
ax2.set_ylim(0, max(te_rmsle) * 1.28)
ax2.set_title('Train vs Test  ×  Before / After — RMSLE 비교',
              fontsize=12, fontweight='bold', pad=14)
ax2.grid(axis='y', linestyle='--', alpha=0.45, zorder=0)
ax2.spines[['top','right']].set_visible(False)

for bar in list(b_tr2) + list(b_te2):
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.005,
             f'{h:.4f}', ha='center', va='bottom',
             fontsize=8.5, fontweight='bold', color='#2c3e50')

# 개선 화살표 (Test: Before → LightGBM)
ax2.annotate('',
    xy=(x[2]+w/2, rmsle_te_lgbm), xytext=(x[0]+w/2, rmsle_te_raw),
    arrowprops=dict(arrowstyle='->', color='#993C1D', lw=1.8,
                    connectionstyle='arc3,rad=0.25'))
ax2.text(1.55, (rmsle_te_raw + rmsle_te_lgbm)/2 + 0.01,
         f'{rmsle_te_lgbm-rmsle_te_raw:.4f}', color='#993C1D',
         fontsize=9, fontweight='bold')

legend_patches2 = [
    plt.Rectangle((0,0),1,1, color=AMB_TR,  label='Train · Before Tuning'),
    plt.Rectangle((0,0),1,1, color=AMB_TR2, label='Train · After Tuning (GBR)'),
    plt.Rectangle((0,0),1,1, color=AMB_TR3, label='Train · After Tuning (LightGBM)'),
    plt.Rectangle((0,0),1,1, color=COR_TE,  label='Test · Before Tuning'),
    plt.Rectangle((0,0),1,1, color=COR_TE2, label='Test · After Tuning (GBR)'),
    plt.Rectangle((0,0),1,1, color=COR_TE3, label='Test · After Tuning (LightGBM)'),
]
ax2.legend(handles=legend_patches2, fontsize=8, loc='upper right',
           ncol=2, framealpha=0.9)

plt.tight_layout()
plt.show()

# ── Fig 3: 실제 vs 예측 흐름 ─────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(14, 5.5))
n = 60
xi = np.arange(n)

ax3.plot(xi, y_te.values[:n],     color='black',   lw=2.5, ls='--', label='실제 주문량')
ax3.plot(xi, p_te_raw[:n],        color='#aaaaaa', lw=1.5, alpha=0.8, label='Before Tuning (GBR)')
ax3.plot(xi, p_te_tuned[:n],      color=TEAL_TE2,  lw=1.8, label='After Tuning (GBR+GridSearch)')
ax3.plot(xi, p_te_lgbm[:n],       color='#E65100', lw=2.5, label='After Tuning (LightGBM, 최종)')

ax3.set_title('실제 배달량 vs 예측 흐름 — Before / After 비교 (Test 앞 60개)',
              fontsize=12, fontweight='bold', pad=15)
ax3.set_xlabel('테스트 샘플 인덱스', fontsize=10)
ax3.set_ylabel('배달 주문량 (건)', fontsize=10)
ax3.grid(True, linestyle='--', alpha=0.45)
ax3.spines[['top','right']].set_visible(False)
ax3.legend(bbox_to_anchor=(1.01, 0.95), loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()

print("\n[개선 요약]")
print(f"  Test R²    : {r2_te_raw:.4f} → {r2_te_lgbm:.4f}  (+{r2_te_lgbm-r2_te_raw:.4f})")
print(f"  Test RMSLE : {rmsle_te_raw:.4f} → {rmsle_te_lgbm:.4f}  ({rmsle_te_lgbm-rmsle_te_raw:.4f})")
print(f"  과적합(R² 격차) : {r2_tr_raw-r2_te_raw:.4f} → {r2_tr_lgbm-r2_te_lgbm:.4f}")

import joblib

# 모델과 피처 리스트를 딕셔너리로 묶어서 저장

pipeline_save_data = {
    'model': lgbm,
    'features': feat_after
}

# 파일로 저장
pipeline_path = 'lgbm_delivery_pipeline.pkl'
joblib.dump(pipeline_save_data, pipeline_path)

print("\n" + "="*50)
print(f" 성공: 모델 및 피처 리스트가 '{pipeline_path}'에 통합 저장되었습니다.")
print("="*50)

# import os
# import joblib
# import numpy as np
# import pandas as pd
# import warnings
# warnings.filterwarnings('ignore')

# # =====================================================================
# # 1. 통합 파이프라인 파일 로드
# # =====================================================================
# pipeline_path = 'lgbm_delivery_pipeline.pkl'

# if not os.path.exists(pipeline_path):
#     raise FileNotFoundError(f"'{pipeline_path}' 파일을 찾을 수 없습니다. 먼저 학습 코드를 실행해 주세요.")

# # 저장했던 딕셔너리 불러오기
# pipeline_data = joblib.load(pipeline_path)
# model = pipeline_data['model']
# trained_features = pipeline_data['features']

# print(" 모델 및 피처 리스트 로드 완료!")
# print(f"학습된 총 피처 개수: {len(trained_features)}개")


# # =====================================================================
# # 2. 신규 데이터 전처리 함수 (학습 당시의 파이프라인 완벽 재현)
# # =====================================================================
# def preprocess_pipeline(df_new, feat_order):
#     df_res = df_new.copy()

#     # 날짜 및 시간 파생 피처
#     df_res['날짜_dt'] = pd.to_datetime(df_res['날짜'])
#     df_res['월'] = df_res['날짜_dt'].dt.month
#     df_res['일'] = df_res['날짜_dt'].dt.day

#     # 피크타임 및 주말 여부 (기존 로직과 동일)
#     # ※ 필요 시 공휴일(Is_Holiday) 리스트 비교 로직을 여기에 추가할 수 있습니다.
#     df_res['Is_Holiday'] = 0  
#     df_res['Is_PeakTime'] = df_res['시간'].apply(lambda x: 1 if (11<=x<=13) or (17<=x<=20) else 0)
#     df_res['Is_Weekend']   = df_res['요일'].isin(['토요일','일요일']).astype(int)
#     df_res['Is_Lunch']     = ((df_res['시간']>=11)&(df_res['시간']<=13)).astype(int)
#     df_res['Is_Dinner']    = ((df_res['시간']>=17)&(df_res['시간']<=20)).astype(int)
#     df_res['Is_LateNight'] = ((df_res['시간']>=21)|(df_res['시간']<=3)).astype(int)

#     # 삼각함수를 이용한 주기성 인코딩 (AI 코드 영역)
#     df_res['시간_sin'] = np.sin(2*np.pi*df_res['시간']/24)
#     df_res['시간_cos'] = np.cos(2*np.pi*df_res['시간']/24)
#     df_res['월_sin']   = np.sin(2*np.pi*df_res['월']/12)
#     df_res['월_cos']   = np.cos(2*np.pi*df_res['월']/12)
    
#     day_map = {'월요일':0,'화요일':1,'수요일':2,'목요일':3,'금요일':4,'토요일':5,'일요일':6}
#     df_res['요일_num'] = df_res['요일'].map(day_map)
#     df_res['요일_sin'] = np.sin(2*np.pi*df_res['요일_num']/7)
#     df_res['요일_cos'] = np.cos(2*np.pi*df_res['요일_num']/7)

#     # 기상 상황 및 활동 지수 파생 변수
#     df_res['기온_강수'] = df_res['기온'] * df_res['강수량']
#     df_res['기온_제곱'] = df_res['기온'] ** 2
#     df_res['강수_있음'] = (df_res['강수량']>0).astype(int)
#     df_res['적설_있음'] = (df_res['적설']>0).astype(int)
#     df_res['쾌적도']   = -np.abs(df_res['기온']-17.5)
#     df_res['Outdoor_Activity_Index'] = df_res['기온']-(df_res['강수량']*2.5)-(df_res['적설']*4.0)

#     # 요일 원-핫 인코딩
#     df_res = pd.get_dummies(df_res, columns=['요일'], drop_first=False)

#     # [핵심] 신규 데이터에 없는 요일 컬럼 자동 생성 및 데이터 정렬
#     for col in feat_order:
#         if col not in df_res.columns:
#             df_res[col] = 0  # 신규 데이터에 없는 요일(예: 수집 안 된 요일)은 0으로 채움

#     # 원본 학습 피처 순서와 완벽히 일치하도록 슬라이싱
#     X_new = df_res[feat_order]
#     return X_new


# # =====================================================================
# # 3. 새로운 데이터 입력 및 예측 테스트
# # =====================================================================
# # 외부 CSV를 읽어오거나 API 데이터를 받아올 때, 아래의 컬럼 구조 형태를 유지해야 합니다.
# new_data = pd.DataFrame([
#     {
#         '날짜': '2026-06-10', '시간': 12, '요일': '수요일',
#         '기온': 26.5, '강수량': 0.0, '적설': 0.0, '시정': 20000
#     },
#     {
#         '날짜': '2026-06-10', '시간': 18, '요일': '수요일', 
#         '기온': 19.0, '강수량': 12.5, '적설': 0.0, '시정': 4500
#     }
# ])

# print("\n 입력 데이터 확인:")
# print(new_data)

# # 전처리 실행
# X_test = preprocess_pipeline(new_data, trained_features)

# # 예측 수행 및 로그 역전환(np.expm1) 적용
# log_preds = model.predict(X_test)
# real_preds = np.expm1(log_preds)

# # 최종 결과 저장 및 출력
# new_data['예측_주문건수'] = np.round(real_preds, 1)

# print("\n" + "="*50)
# print("  최종 예측 결과")
# print("="*50)
# print(new_data[['날짜', '시간', '요일', '기온', '강수량', '예측_주문건수']])
# print("="*50)