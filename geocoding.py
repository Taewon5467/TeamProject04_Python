# 네이버 API 좌표 변환
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, FILE_RAW, FILE_GEO_CACHED, MAX_THREADS


def _fetch_single(index, address):
    url = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY":    NAVER_CLIENT_SECRET,
    }
    try:
        resp = requests.get(url, headers=headers, params={"query": address}, timeout=5)
        addrs = resp.json().get("addresses", [])
        if addrs:
            return index, float(addrs[0]["y"]), float(addrs[0]["x"])
    except Exception:
        pass
    return index, None, None


def load_with_geocoding():
    if os.path.exists(FILE_GEO_CACHED):
        print(f"📦 캐시 파일 로드: {FILE_GEO_CACHED}")
        return _read_csv(FILE_GEO_CACHED)

    print(f"🔍 원본 파일 로드: {FILE_RAW}")
    df = _read_csv(FILE_RAW)

    if '주소' not in df.columns or '가게명' not in df.columns:
        raise ValueError("❌ CSV에 '주소' 또는 '가게명' 컬럼이 없습니다.")

    df['정제된_주소'] = df['주소'].astype(str).str.split().str[:5].str.join(' ')

    print(f"🚀 병렬 좌표 변환 시작 (스레드 {MAX_THREADS}개)...")
    results = [None] * len(df)
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(_fetch_single, idx, addr): idx
                   for idx, addr in enumerate(df['정제된_주소'].tolist())}
        done = 0
        for future in as_completed(futures):
            idx, lat, lon = future.result()
            results[idx] = (lat, lon)
            done += 1
            if done % 50 == 0 or done == len(df):
                print(f"   ➔ {done}/{len(df)} ({done/len(df)*100:.1f}%)")

    df['lat'] = [r[0] for r in results]
    df['lon'] = [r[1] for r in results]
    df_clean = df.dropna(subset=['lat', 'lon']).copy()

    if df_clean.empty:
        raise RuntimeError("❌ 좌표 변환 성공 건 없음. API 키를 확인하세요.")

    df_clean.to_csv(FILE_GEO_CACHED, index=False, encoding='utf-8-sig')
    print(f"💾 캐시 저장: {FILE_GEO_CACHED}")
    return df_clean


def _read_csv(path):
    for enc in ('utf-8', 'utf-8-sig', 'cp949'):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"❌ CSV를 읽을 수 없습니다: {path}")