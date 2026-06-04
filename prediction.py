# ML 예측 + 로봇 배분 + 결과 저장
import math
import os
import joblib
import numpy as np
import pandas as pd
from config import PIPELINE_PATH, OUTPUT_SIM_DIR


def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        print(f"⚠️  파이프라인 파일 없음: {PIPELINE_PATH}")
        return None, None
    data     = joblib.load(PIPELINE_PATH)
    model    = data['model']
    features = data['features']
    print(f"✅ 모델 로드 완료 — 피처 {len(features)}개")
    return model, features


def preprocess_for_prediction(df_new, feat_order, target_weekday, is_holiday=0):
    df = df_new.copy()
    df['날짜_dt'] = pd.to_datetime(df['날짜'])
    df['월']      = df['날짜_dt'].dt.month
    df['일']      = df['날짜_dt'].dt.day
    df['요일']    = target_weekday

    df['Is_Holiday']  = is_holiday
    df['Is_PeakTime'] = df['시간'].apply(lambda x: 1 if (11<=x<=13) or (17<=x<=20) else 0)
    df['Is_Weekend']  = df['요일'].isin(['토요일','일요일']).astype(int)
    df['Is_Lunch']    = ((df['시간']>=11)&(df['시간']<=13)).astype(int)
    df['Is_Dinner']   = ((df['시간']>=17)&(df['시간']<=20)).astype(int)
    df['Is_LateNight']= ((df['시간']>=21)|(df['시간']<=3)).astype(int)

    df['시간_sin'] = np.sin(2*np.pi*df['시간']/24)
    df['시간_cos'] = np.cos(2*np.pi*df['시간']/24)
    df['월_sin']   = np.sin(2*np.pi*df['월']/12)
    df['월_cos']   = np.cos(2*np.pi*df['월']/12)

    day_map = {'월요일':0,'화요일':1,'수요일':2,'목요일':3,'금요일':4,'토요일':5,'일요일':6}
    df['요일_num'] = df['요일'].map(day_map)
    df['요일_sin'] = np.sin(2*np.pi*df['요일_num']/7)
    df['요일_cos'] = np.cos(2*np.pi*df['요일_num']/7)

    df['기온_강수']              = df['기온'] * df['강수량']
    df['기온_제곱']              = df['기온'] ** 2
    df['강수_있음']              = (df['강수량']>0).astype(int)
    df['적설_있음']              = (df['적설']>0).astype(int)
    df['쾌적도']                 = -np.abs(df['기온']-17.5)
    df['Outdoor_Activity_Index'] = df['기온'] - df['강수량']*2.5 - df['적설']*4.0

    df = pd.get_dummies(df, columns=['요일'], drop_first=False)
    for col in feat_order:
        if col not in df.columns:
            df[col] = 0
    return df[feat_order]


def run_simulation(model, trained_features, target_date, weekday, temp, rain, snow, vis, is_holiday=0):
    df_sim = pd.DataFrame({
        '날짜':   [target_date] * 24,
        '시간':   list(range(24)),
        '기온':   [temp]  * 24,
        '강수량': [rain]  * 24,
        '적설':   [snow]  * 24,
        '시정':   [vis]   * 24,
    })
    X_sim       = preprocess_for_prediction(df_sim, trained_features, weekday, is_holiday)
    predictions = np.expm1(model.predict(X_sim))
    return pd.Series(predictions, index=range(24), name='predicted_orders')


def assign_robots(spots, hourly_demand):
    peak_demand = hourly_demand.max()
    avg_demand  = hourly_demand.mean()
    for spot in spots:
        ratio               = spot['store_ratio'] / 100
        spot['robots_peak'] = math.ceil(peak_demand * ratio)  # 나눗셈 제한 없음
        spot['robots_avg']  = math.ceil(avg_demand  * ratio)
    return spots


def save_simulation_result(hourly_demand, target_date, weekday):
    os.makedirs(OUTPUT_SIM_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_SIM_DIR, f"simulation_{target_date}_{weekday}.csv")
    hourly_demand.to_csv(path, header=['predicted_orders'], encoding='utf-8-sig')
    print(f"💾 시뮬레이션 결과 저장: {path}")