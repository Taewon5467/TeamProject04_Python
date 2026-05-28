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