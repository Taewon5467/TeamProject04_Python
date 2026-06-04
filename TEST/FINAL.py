import pandas as pd
import requests
import json
import numpy as np
import math
import os
import joblib  # 파이프라인 로드를 위해 추가
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import haversine_distances
from scipy.cluster.hierarchy import linkage, fcluster, cophenet 
from scipy.spatial.distance import squareform
from scipy.spatial import ConvexHull, QhullError

file_name = r'DataSet\상록구 본오동_통합_가게정보.csv'
geo_saved_file_name = r'DataSet\상록구 본오동_통합_가게정보_좌표추가.csv'
pipeline_path = 'lgbm_delivery_pipeline.pkl'  # ML 모델 파이프라인 파일

from dotenv import load_dotenv
load_dotenv()

NAVER_CLIENT_ID     = os.environ.get("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")

if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
    raise EnvironmentError("❌ .env 파일에 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET을 입력해 주세요.")

# =========================================================================
# [신규 추가] ML 모델용 파이프라인 로드 및 전처리 함수
# =========================================================================
print("--- [머신러닝 예측 모델 파이프라인 로드] ---")
if not os.path.exists(pipeline_path):
    print(f"❌ '{pipeline_path}' 파일이 없습니다. DataModel 코드를 먼저 실행하여 모델을 저장해 주세요.")
    model_loaded = False
else:
    pipeline_data = joblib.load(pipeline_path)
    model = pipeline_data['model']
    trained_features = pipeline_data['features']
    model_loaded = True
    print(f"✅ 기계학습 모델 및 피처 {len(trained_features)}개 로드 완료.\n")


def preprocess_for_prediction(df_new, feat_order, target_weekday):
    """신규 환경 변수 데이터를 학습 데이터 포맷과 일치시키는 전처리 함수"""
    df_res = df_new.copy()

    # 날짜 분할 및 기본 피처 변환
    df_res['날짜_dt'] = pd.to_datetime(df_res['날짜'])
    df_res['월'] = df_res['날짜_dt'].dt.month
    df_res['일'] = df_res['날짜_dt'].dt.day
    df_res['요일'] = target_weekday

    # 피크타임 및 상태 변수 정의
    df_res['Is_Holiday'] = 0  # 시뮬레이션 상 평일 기준 (필요시 1 설정)
    df_res['Is_PeakTime'] = df_res['시간'].apply(lambda x: 1 if (11<=x<=13) or (17<=x<=20) else 0)
    df_res['Is_Weekend']   = df_res['요일'].isin(['토요일','일요일']).astype(int)
    df_res['Is_Lunch']     = ((df_res['시간']>=11)&(df_res['시간']<=13)).astype(int)
    df_res['Is_Dinner']    = ((df_res['시간']>=17)&(df_res['시간']<=20)).astype(int)
    df_res['Is_LateNight'] = ((df_res['시간']>=21)|(df_res['시간']<=3)).astype(int)

    # 주기성 피처 엔지니어링 (sin/cos 변환)
    df_res['시간_sin'] = np.sin(2*np.pi*df_res['시간']/24)
    df_res['시간_cos'] = np.cos(2*np.pi*df_res['시간']/24)
    df_res['월_sin']   = np.sin(2*np.pi*df_res['월']/12)
    df_res['월_cos']   = np.cos(2*np.pi*df_res['월']/12)
    
    day_map = {'월요일':0,'화요일':1,'수요일':2,'목요일':3,'금요일':4,'토요일':5,'일요일':6}
    df_res['요일_num'] = df_res['요일'].map(day_map)
    df_res['요일_sin'] = np.sin(2*np.pi*df_res['요일_num']/7)
    df_res['요일_cos'] = np.cos(2*np.pi*df_res['요일_num']/7)

    # 복합 기상 영향 지수 산출
    df_res['기온_강수'] = df_res['기온'] * df_res['강수량']
    df_res['기온_제곱'] = df_res['기온'] ** 2
    df_res['강수_있음'] = (df_res['강수량']>0).astype(int)
    df_res['적설_있음'] = (df_res['적설']>0).astype(int)
    df_res['쾌적도']   = -np.abs(df_res['기온']-17.5)
    df_res['Outdoor_Activity_Index'] = df_res['기온']-(df_res['강수량']*2.5)-(df_res['적설']*4.0)

    # 요일 원-핫 인코딩 적용
    df_res = pd.get_dummies(df_res, columns=['요일'], drop_first=False)

    # 누락된 학습 요일 변수 존재 시 0으로 보정
    for col in feat_order:
        if col not in df_res.columns:
            df_res[col] = 0

    # 학습 순서대로 정렬 및 슬라이싱 후 반환
    return df_res[feat_order]


# =========================================================================
# 1. 환경 피처 데이터 입력 및 시뮬레이션 설정
# =========================================================================
print("--- [시뮬레이션 기상 조건 입력] ---")
target_date = "2026-06-15"  # 예측 타겟 날짜 설정

if model_loaded:
    weekday_input = input("1. 분석할 요일을 입력하세요 (예: 월요일, 토요일 등): ").strip()
    if not weekday_input: weekday_input = "월요일"
    
    try:
        temp_input = float(input("2. 시뮬레이션 온도(°C)를 입력하세요 (예: 22.5): "))
    except ValueError: temp_input = 17.5
        
    try:
        rain_input = float(input("3. 시뮬레이션 강수량(mm)을 입력하세요 (없으면 0): "))
    except ValueError: rain_input = 0.0
        
    try:
        snow_input = float(input("4. 시뮬레이션 적설량(cm)을 입력하세요 (없으면 0): "))
    except ValueError: snow_input = 0.0
        
    try:
        vis_input = float(input("5. 시뮬레이션 시정 수치를 입력하세요 (예: 20000): "))
    except ValueError: vis_input = 20000.0
else:
    print("⚠️ 머신러닝 파이프라인이 없어 기본 시뮬레이션 값으로 진행합니다.")
    weekday_input = "월요일"


# =========================================================================
# 2. 가게 정보 로드 및 네이버 API 병렬 좌표 변환
# =========================================================================
if os.path.exists(geo_saved_file_name):
    print(f"\n📦 기존에 변환 완료된 좌표 파일({geo_saved_file_name})을 발견하여 API 호출 없이 로드합니다.")
    try:
        df_clean = pd.read_csv(geo_saved_file_name, encoding='utf-8')
    except UnicodeDecodeError:
        df_clean = pd.read_csv(geo_saved_file_name, encoding='cp949')
else:
    print(f"\n🔍 최초 실행 또는 좌표 파일이 없어 원본 파일({file_name})을 로드합니다.")
    try:
        df = pd.read_csv(file_name, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_name, encoding='cp949')

    if '주소' not in df.columns or '가게명' not in df.columns:
        print("❌ CSV 파일에 '주소' 또는 '가게명' 컬럼이 없습니다. 프로그램을 종료합니다.")
        exit()

    df['정제된_주소'] = df['주소'].astype(str).str.split(' ').str[:5].str.join(' ')

    def fetch_lat_lon(index, address):
        url = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
            "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET
        }
        params = { "query": address }
        try:
            response = requests.get(url, headers=headers, params=params, timeout=5)
            res_json = response.json()
            if 'addresses' in res_json and res_json['addresses']:
                y = float(res_json['addresses'][0]['y']) 
                x = float(res_json['addresses'][0]['x']) 
                return index, y, x
        except Exception as e:
            pass
        return index, None, None

    print("🚀 멀티스레딩 기반 병렬 처리로 네이버 API 좌표 변환을 시작합니다...")
    results = [None] * len(df)
    address_list = df['정제된_주소'].tolist()
    max_threads = 10 
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(fetch_lat_lon, idx, addr): idx for idx, addr in enumerate(address_list)}
        completed_count = 0
        for future in as_completed(futures):
            idx, lat, lon = future.result()
            results[idx] = (lat, lon)
            completed_count += 1
            if completed_count % 50 == 0 or completed_count == len(df):
                print(f"   ➔ 좌표 변환 진행률: {completed_count}/{len(df)} 건 완료 ({(completed_count/len(df))*100:.1f}%)")

    df['lat'] = [r[0] for r in results]
    df['lon'] = [r[1] for r in results]
    df_clean = df.dropna(subset=['lat', 'lon']).copy()
    
    if not df_clean.empty:
        df_clean.to_csv(geo_saved_file_name, index=False, encoding='utf-8-sig')
        print(f"💾 좌표 변환 데이터가 정상 보관되었습니다: {geo_saved_file_name}")

# =========================================================================
# 3. DCF 기반 거리-시간 변환 및 군집화
# =========================================================================
if not df_clean.empty:
    target_time_min = 10      
    target_time_hours = target_time_min / 60  
    
    SF = 15.0  
    CF = 1.35  
    DF = 2.22  
    
    DCF = (CF * DF) / SF
    max_distance_km = target_time_hours / DCF
    
    df_clean['lat_rad'] = np.radians(df_clean['lat'])
    df_clean['lon_rad'] = np.radians(df_clean['lon'])
    coords_rad = df_clean[['lat_rad', 'lon_rad']].values
    
    dist_matrix = haversine_distances(coords_rad) * 6371.0088
    dist_array = squareform(dist_matrix)
    Z = linkage(dist_array, method='complete')
    
    df_clean['cluster'] = fcluster(Z, t=max_distance_km, criterion='distance')
    
    locations = df_clean[['가게명', 'lat', 'lon', 'cluster']].to_dict('records')
    locations_json = json.dumps(locations, ensure_ascii=False)
    
    spots = []
    unique_clusters = set(df_clean['cluster'])
    
    for cluster_id in unique_clusters:
        cluster_df = df_clean[df_clean['cluster'] == cluster_id]
        
        spot_lat = cluster_df['lat'].mean()
        spot_lon = cluster_df['lon'].mean()
        store_count = len(cluster_df)
        
        points = cluster_df[['lat', 'lon']].values
        hull_coordinates = []
        
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                for vertex in hull.vertices:
                    hull_coordinates.append({
                        "lat": float(points[vertex][0]),
                        "lon": float(points[vertex][1])
                    })
            except QhullError:
                hull_coordinates = [{"lat": float(p[0]), "lon": float(p[1])} for p in points]
        else:
            hull_coordinates = [{"lat": float(p[0]), "lon": float(p[1])} for p in points]
        
        spots.append({
            "cluster_id": int(cluster_id),
            "lat": spot_lat,
            "lon": spot_lon,
            "count": store_count,
            "hull": hull_coordinates
        })

    # =========================================================================
    # 4. [수정] 머신러닝 예측 기반 시뮬레이션 로봇 수요 산출 로직
    # =========================================================================
    if model_loaded:
        # 0시~23시까지의 기상 시뮬레이션 프레임워크 구축
        sim_hours = list(range(24))
        df_sim = pd.DataFrame({
            '날짜': [target_date] * 24,
            '시간': sim_hours,
            '기온': [temp_input] * 24,
            '강수량': [rain_input] * 24,
            '적설': [snow_input] * 24,
            '시정': [vis_input] * 24
        })

        # 전처리 매핑 적용 및 예측
        X_sim = preprocess_for_prediction(df_sim, trained_features, weekday_input)
        log_predictions = model.predict(X_sim)
        sim_predictions = np.expm1(log_predictions)  # 로그 역변환 복원

        # 시간대별 예측 결과 매핑 구조화
        hourly_demand = pd.Series(sim_predictions, index=sim_hours)
        
        peak_hour = hourly_demand.idxmax()
        peak_demand = hourly_demand.max()
        avg_demand = hourly_demand.mean() 
        
        print(f"\n--- [🤖 기계학습 예측 기반 로봇 대수 시뮬레이션 결과 ({weekday_input})] ---")
        print(f"■ 입력 조건   : 온도 {temp_input}°C / 강수 {rain_input}mm / 적설 {snow_input}cm / 시정 {vis_input}")
        print(f"■ 최고 피크 타임: {peak_hour}시 (예측 총량: {peak_demand:.1f}건)")
        print(f"■ 일일 평시 평균: {avg_demand:.1f}건")
        
        total_stores = len(df_clean)
        for spot in spots:
            store_ratio = spot['count'] / total_stores
            spot_peak_demand = peak_demand * store_ratio
            spot_avg_demand = avg_demand * store_ratio
            
            spot['store_ratio'] = round(store_ratio * 100, 1)
            # 배달 로봇 1대당 2건 처리 능력 기준 올림 연산 처리
            spot['robots_peak'] = int(math.ceil(spot_peak_demand / 2.0))
            spot['robots_avg'] = int(math.ceil(spot_avg_demand / 2.0))
    else:
        print("⚠️ 학습 모델을 로드할 수 없어 기본 대체 텍스트로 표기됩니다.")
        for spot in spots:
            spot['store_ratio'], spot['robots_peak'], spot['robots_avg'] = 0, "?", "?"

    spots_json = json.dumps(spots, ensure_ascii=False)
    center_lat = df_clean['lat'].mean()
    center_lon = df_clean['lon'].mean()

    # =========================================================================
    # 5. HTML 파일 생성
    # =========================================================================
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>네이버 지도 - 로봇 최적 거점 분석 (ML 예측 모드)</title>
    <script type="text/javascript" src="https://oapi.map.naver.com/openapi/v3/maps.js?ncpKeyId={NAVER_CLIENT_ID}"></script>
    <style>
        html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; }}
        #map {{ width: 100%; height: 100vh; }}
    </style>
</head>
<body>
<div id="map"></div>
<script>
    var map = new naver.maps.Map('map', {{
        center: new naver.maps.LatLng({center_lat}, {center_lon}),
        zoom: 14
    }});

    // 1. 가게 마커 표시
    var locations = {locations_json};
    for (var i = 0; i < locations.length; i++) {{
        var marker = new naver.maps.Marker({{
            position: new naver.maps.LatLng(locations[i].lat, locations[i].lon),
            map: map,
            icon: {{
                content: '<div style="width:6px; height:6px; background:#007bef; border-radius:50%; border:1px solid white;"></div>',
                anchor: new naver.maps.Point(3, 3)
            }}
        }});
    }}

    // 2. DCF 기준 거점 다각형 표시
    var spots = {spots_json};
    for (var j = 0; j < spots.length; j++) {{
        var hullData = spots[j].hull;
        var polygonPath = [];
        for (var k = 0; k < hullData.length; k++) {{
            polygonPath.push(new naver.maps.LatLng(hullData[k].lat, hullData[k].lon));
        }}

        if (polygonPath.length >= 3) {{
            var polygon = new naver.maps.Polygon({{
                map: map, paths: [polygonPath],
                fillColor: '#10b981', fillOpacity: 0.15,
                strokeColor: '#059669', strokeOpacity: 0.6, strokeWeight: 2,
                clickable: false
            }});
        }} else if (polygonPath.length === 2) {{
            var polyline = new naver.maps.Polyline({{
                map: map, path: polygonPath,
                strokeColor: '#059669', strokeOpacity: 0.6, strokeWeight: 2
            }});
        }}

        // 3. 거점 정보 마커 표시 (ML 예측 모델 바인딩 데이터 반영)
        var spotMarker = new naver.maps.Marker({{
            position: new naver.maps.LatLng(spots[j].lat, spots[j].lon),
            map: map,
            icon: {{
                content: '<div style="background:#0f172a; color:white; padding:5px 8px; border-radius:6px; font-family:sans-serif; border:1px solid #334155; box-shadow: 0px 4px 6px rgba(0,0,0,0.4); white-space:nowrap; text-align:center; line-height:1.4;">' +
                         '<div style="font-size:11px; font-weight:bold; color:#38bdf8; margin-bottom:2px;">거점 #' + (j+1) + ' (' + spots[j].count + '개 소속)</div>' +
                         '<div style="font-size:10px; color:#fde047; font-weight:bold;">🤖 피크 ' + spots[j].robots_peak + '대 | 평시 ' + spots[j].robots_avg + '대</div>' +
                         '</div>',
                anchor: new naver.maps.Point(40, 15)
            }}
        }});
    }}
</script>
</body>
</html>
"""

    output_file = 'naver_store_map.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n🎯 [성공] ML 시뮬레이션 기반 HTML 지도 생성 완료: {output_file}")

    # =========================================================================
    # 6. 모델 평가
    # =========================================================================
    print("\n=======================================================")
    print("📊 [평가 1] 코페네틱 상관계수 (CPCC) 분석")
    print("=======================================================")
    c, coph_dists = cophenet(Z, dist_array)
    print(f"■ 산출된 코페네틱 상관계수: {c:.4f}")
    
    print("\n=======================================================")
    print("🛠 [평가 2] 물리적 거리 제약(10분 배달 보장) 실효성 검증")
    print("=======================================================")
    violation_count = 0
    max_observed_dist_km = 0
    
    for cluster_id in unique_clusters:
        cluster_coords = df_clean[df_clean['cluster'] == cluster_id][['lat_rad', 'lon_rad']].values
        if len(cluster_coords) > 1:
            pair_dists = haversine_distances(cluster_coords) * 6371.0088
            cluster_max_dist = pair_dists.max()
            if cluster_max_dist > max_observed_dist_km:
                max_observed_dist_km = cluster_max_dist
            if cluster_max_dist > max_distance_km:
                violation_count += 1
                
    print(f"■ 목표 제한 거리 (Threshold) : {max_distance_km:.4f} km")
    print(f"■ 생성된 군집 내 실제 최대 거리: {max_observed_dist_km:.4f} km")
    print(f"■ 제한 거리를 초과한 군집 수   : {violation_count}개")
    print("=======================================================\n")

else:
    print("❌ 좌표 변환된 데이터가 없습니다.")
    
#python -m http.server 8000
#http://localhost:8000/naver_store_map.html