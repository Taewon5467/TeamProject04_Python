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
    
    data = joblib.load(PIPELINE_PATH)
    print(type(data))
    print("PIPELINE_PATH =", PIPELINE_PATH)
    
    # 딕셔너리 구조가 아니라 바로 Pipeline 객체가 저장된 경우를 대비한 방어 코드
    if isinstance(data, dict):
        model = data['model']
        features = data['features']
    else:
        model = data
        features = None # 혹은 필요한 피처 리스트
        
    print(f"✅ 모델 로드 완료")
    return model, features


def preprocess_for_prediction(df_new, feat_order, target_weekday, is_holiday=0):
    df = df_new.copy()

    # 날짜 파생
    df['날짜_dt'] = pd.to_datetime(df['날짜'])

    df['월'] = df['날짜_dt'].dt.month
    df['일'] = df['날짜_dt'].dt.day

    df['시간'] = df['시간'].astype(int)

    # 주말 여부
    df['주말'] = (
        df['요일'].isin(['토요일', '일요일'])
    ).astype(int)

    # 계절 생성
    def season(month):
        if month in [3, 4, 5]:
            return "봄"
        elif month in [6, 7, 8]:
            return "여름"
        elif month in [9, 10, 11]:
            return "가을"
        else:
            return "겨울"

    df['계절'] = df['월'].apply(season)

    # 모델이 요구하는 컬럼만 반환
    required_cols = [
        '요일',
        '시도',
        '시군구',
        '계절',
        '시간',
        '기온',
        '강수량',
        '적설',
        '시정',
        '연도',
        '월',
        '일',
        '주말'
    ]

    # 디버깅
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"누락 컬럼: {missing}")

    return df[required_cols]


def run_simulation(pipeline, feat_order, target_date, weekday,
                   temp, rain, snow, vis, is_holiday=0):
    print("features =", feat_order)
    print("weekday =", weekday)
    # 24시간 예측을 위한 베이스 데이터 생성
    df_sim = pd.DataFrame({
        '날짜': [target_date] * 24,
        '시간': list(range(24)),
        '기온': [temp] * 24,
        '강수량': [rain] * 24,
        '적설': [snow] * 24,
        '시정': [vis] * 24,
        '요일': [weekday] * 24,
        '시도': ['경기도'] * 24,
        '시군구': ['안산시 상록구'] * 24,
        '연도': [int(target_date.split('-')[0])] * 24
    })

    X_sim = preprocess_for_prediction(
        df_sim,
        feat_order,
        weekday,
        is_holiday
    )
    print("\n===== 예측 입력 데이터 =====")
    (X_sim[['요일', '주말']].head())

    predictions = pipeline.predict(X_sim)
    
    print("\n===== 예측 결과 일부 =====")
    print(predictions[:10])

    return pd.Series(
        predictions,
        index=range(24),
        name='predicted_orders'
    )


def assign_robots(spots, hourly_demand):
    peak_demand = hourly_demand.max()
    avg_demand = hourly_demand.mean()

    print("\n===== 로봇 배치 계산 =====")
    print(f"피크 주문량 : {peak_demand:.2f}")
    print(f"평균 주문량 : {avg_demand:.2f}")

    for spot in spots:
        peak_calc = (peak_demand * (spot['store_ratio'] / 100) * 0.3)
        avg_calc = (avg_demand * (spot['store_ratio'] / 100) * 0.3)
        spot['robots_peak'] = str(max(1, math.ceil(peak_calc)))
        spot['robots_avg'] = str(max(1, math.ceil(avg_calc)))

        print(
            f"거점 | "
            f"비율:{spot['store_ratio']:.2f}% | "
            f"피크계산:{peak_calc:.2f} -> {spot['robots_peak']}대 | "
            f"평균계산:{avg_calc:.2f} -> {spot['robots_avg']}대"
        )

    return spots


def save_simulation_result(hourly_demand, target_date, weekday):
    os.makedirs(OUTPUT_SIM_DIR, exist_ok=True)
    df_res = hourly_demand.to_frame().reset_index()
    df_res.columns = ['시간', '예측주문건수']
    df_res['날짜'] = target_date
    df_res['요일'] = weekday
    df_res = df_res[['날짜', '요일', '시간', '예측주문건수']]
    
    out_path = os.path.join(OUTPUT_SIM_DIR, f"simulation_{target_date}.csv")
    df_res.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"💾 시뮬레이션 결과 저장 완료: {out_path}")