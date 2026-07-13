import pandas as pd
import requests
import json
import numpy as np
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed # 병렬 처리를 위해 추가
from sklearn.metrics.pairwise import haversine_distances
from scipy.cluster.hierarchy import linkage, fcluster, cophenet 
from scipy.spatial.distance import squareform
from scipy.spatial import ConvexHull, QhullError
import sys

# =========================================================================
# 파일 경로 및 API 키 설정
# =========================================================================
file_name = r'DataSet\상록구 본오동_통합_가게정보.csv'
geo_saved_file_name = r'DataSet\상록구 상록구 통합 가게정보_좌표추가.csv'
ml_file_name = r'DataSet\ansan_delivery_2021_clean.csv'

NAVER_CLIENT_ID = 'hymjc5hhjr'
NAVER_CLIENT_SECRET = 'DCpgXMURtZgZ2HTQqywyvVxaoYj1A9Rx6M5G8ZdC'

# =========================================================================
# 1. 요일 입력 및 주문 데이터 사전 필터링
# =========================================================================
print("--- [데이터 로딩 및 분석 옵션 선택] ---")
df_orders = None
target_weekday = "전체 요일"

try:
    df_orders = pd.read_csv(ml_file_name, encoding='utf-8')
    if '요일' in df_orders.columns:
        unique_days = df_orders['요일'].dropna().unique().tolist()
        week_order = {"월요일": 1, "화요일": 2, "수요일": 3, "목요일": 4, "금요일": 5, "토요일": 6, "일요일": 7}
        sorted_days = sorted(unique_days, key=lambda x: week_order.get(x, 8))
        days_str = ", ".join(sorted_days)
        
        print(f"▶ 현재 데이터셋에 있는 요일: [ {days_str} ]")
        target_weekday = input("분석할 요일을 위 목록에서 똑같이 입력하세요 (전체 요일을 분석하려면 그냥 Enter): ").strip()
        
        if target_weekday:
            if target_weekday in unique_days:
                df_orders = df_orders[df_orders['요일'] == target_weekday]
                print(f"✅ '{target_weekday}' 주문 데이터만 필터링하여 분석을 시작합니다.\n")
            else:
                print(f"⚠️ 데이터셋에 '{target_weekday}' 데이터가 없습니다. 전체 요일 기준으로 진행합니다.\n")
                target_weekday = "전체 요일"
        else:
            print("✅ 전체 요일 데이터를 기준으로 분석을 시작합니다.\n")
            target_weekday = "전체 요일"
    else:
        print("⚠️ 데이터셋에 '요일' 컬럼이 없습니다. 전체 요일 데이터를 기준으로 진행합니다.\n")
        
except FileNotFoundError:
    print(f"❌ '{ml_file_name}' 파일을 찾을 수 없어 주문량 기반 수요 예측은 생략됩니다.\n")


# =========================================================================
# 2. 가게 정보 로드 및 네이버 API 병렬 좌표 변환
# =========================================================================
if os.path.exists(geo_saved_file_name):
    print(f"📦 기존에 변환 완료된 좌표 파일({geo_saved_file_name})을 발견하여 API 호출 없이 로드합니다.")
    try:
        df_clean = pd.read_csv(geo_saved_file_name, encoding='utf-8')
    except UnicodeDecodeError:
        df_clean = pd.read_csv(geo_saved_file_name, encoding='cp949')
else:
    print(f"🔍 최초 실행 또는 좌표 파일이 없어 원본 파일({file_name})을 로드합니다.")
    try:
        df = pd.read_csv(file_name, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_name, encoding='cp949')

    if '주소' not in df.columns or '가게명' not in df.columns:
        print("❌ CSV 파일에 '주소' 또는 '가게명' 컬럼이 없습니다. 프로그램을 종료합니다.")
        exit()

    df['정제된_주소'] = df['주소'].astype(str).str.split(' ').str[:5].str.join(' ')

    # 단일 주소를 변환하는 핵심 함수
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
    
    # 결과 데이터를 임시 매핑할 딕셔너리 및 리스트 초기화
    results = [None] * len(df)
    address_list = df['정제된_주소'].tolist()
    
    # [병렬 처리 핵심 부] max_workers 속도를 높이려면 스레드 수를 조정(예: 10~20)
    # 네이버 API 요금제 및 Rate Limit에 맞춰 안전하게 10개 스레드로 동시 요청 처리
    max_threads = 10 
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # 워커스레드에 태스크 할당
        futures = {executor.submit(fetch_lat_lon, idx, addr): idx for idx, addr in enumerate(address_list)}
        
        # 진행 상황 표기 및 데이터 기록
        completed_count = 0
        for future in as_completed(futures):
            idx, lat, lon = future.result()
            results[idx] = (lat, lon)
            completed_count += 1
            if completed_count % 50 == 0 or completed_count == len(df):
                print(f"   ➔ 좌표 변환 진행률: {completed_count}/{len(df)} 건 완료 ({(completed_count/len(df))*100:.1f}%)")

    # 병렬 처리 결과 프레임에 반영
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
    # 4. 시간대별 로봇 수요 산출 로직
    # =========================================================================
    if df_orders is not None:
        hourly_demand = df_orders.groupby('시간')['주문건수'].mean()
        
        if hourly_demand.empty:
            print("❌ 조건에 맞는 주문 데이터가 없어 로봇 수요를 산출할 수 없습니다.")
            for spot in spots:
                spot['store_ratio'], spot['robots_peak'], spot['robots_avg'] = 0, "?", "?"
        else:
            peak_hour = hourly_demand.idxmax()
            peak_demand = hourly_demand.max()
            avg_demand = hourly_demand.mean() 
            
            print(f"\n--- [거점별 로봇 대수 산출 완료 ({target_weekday})] ---")
            print(f"■ 전체 피크 타임: {peak_hour}시 (평균 {peak_demand:.1f}건) / 평시 평균: {avg_demand:.1f}건")
            
            total_stores = len(df_clean)
            for spot in spots:
                store_ratio = spot['count'] / total_stores
                spot_peak_demand = peak_demand * store_ratio
                spot_avg_demand = avg_demand * store_ratio
                
                spot['store_ratio'] = round(store_ratio * 100, 1)
                spot['robots_peak'] = int(math.ceil(spot_peak_demand / 2.0))
                spot['robots_avg'] = int(math.ceil(spot_avg_demand / 2.0))
    else:
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
    <title>네이버 지도 - 로봇 최적 거점 분석</title>
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

        // 3. 거점 정보 마커 표시
        var spotMarker = new naver.maps.Marker({{
            position: new naver.maps.LatLng(spots[j].lat, spots[j].lon),
            map: map,
            icon: {{
                content: '<div style="background:#10b981; color:white; padding:8px 12px; border-radius:6px; font-family:sans-serif; border:2px solid white; box-shadow: 0px 3px 6px rgba(0,0,0,0.4); white-space:nowrap; text-align:center; line-height:1.5;">' +
                         '<div style="font-size:18px; font-weight:bold; margin-bottom:3px;">거점 #' + (j+1) + ' (' + spots[j].count + '개)</div>' +
                         '<div style="font-size:16px; color:#ffeb3b; font-weight:bold;">피크 ' + spots[j].robots_peak + '대 | 평시 ' + spots[j].robots_avg + '대</div>' +
                         '</div>',
                anchor: new naver.maps.Point(55, 22)
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

    print(f"\n🎯 [성공] HTML 파일 생성 완료: {output_file}")

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