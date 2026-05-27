import pandas as pd
import requests
import json
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from scipy.cluster.hierarchy import linkage, fcluster, cophenet 
from scipy.spatial.distance import squareform
from scipy.spatial import ConvexHull, QhullError

# 1. 파일 로드 및 주소 정제
file_name = r'상록구 통합 가게정보.csv'

try:
    df = pd.read_csv(file_name, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_name, encoding='cp949')

if '주소' not in df.columns or '가게명' not in df.columns:
    print("❌ CSV 파일에 '주소' 또는 '가게명' 컬럼이 없습니다. 컬럼명을 확인해 주세요.")
    print(f"현재 컬럼명: {list(df.columns)}")
    exit()

df['정제된_주소'] = df['주소'].astype(str).str.split(' ').str[:5].str.join(' ')

# 네이버 Cloud API 키 설정
NAVER_CLIENT_ID = 'hymjc5hhjr'
NAVER_CLIENT_SECRET = 'DCpgXMURtZgZ2HTQqywyvVxaoYj1A9Rx6M5G8ZdC'

def get_naver_lat_lon(address):
    url = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET
    }
    params = { "query": address }
    try:
        response = requests.get(url, headers=headers, params=params)
        res_json = response.json()
        if 'addresses' in res_json and res_json['addresses']:
            y = float(res_json['addresses'][0]['y']) 
            x = float(res_json['addresses'][0]['x']) 
            return pd.Series([y, x])
    except Exception as e:
        print(f" 에러 발생: {e}")
    return pd.Series([None, None])

print("네이버 API로 주소를 좌표로 변환하고 있습니다...")
df[['lat', 'lon']] = df['정제된_주소'].apply(get_naver_lat_lon)
df_clean = df.dropna(subset=['lat', 'lon']).copy()

if not df_clean.empty:

    #거리-시간 변환 계수(DCF) 모델 적용
    target_time_min = 10      # 배송 제한 시간 (10분)
    target_time_hours = target_time_min / 60  # 시간 단위 변환 (1/6 시간)
    
    SF = 15.0  #로봇의 이상적인 주행 속도 (km/h)
    CF = 1.35  #도시 도로망 우회 계수 (직선거리 대비 실도로 거리 비율)
    DF = 2.22  #신호 대기 및 돌발 정체 지연 계수
    
    # 2) DCF 계산 (단위: hours / km) -> 직선거리당 소요되는 실제 시간
    DCF = (CF * DF) / SF
    
    # 3) DCF를 이용한 최대 허용 직선거리(Euclidean/Haversine Threshold) 도출
    max_distance_km = target_time_hours / DCF
    
    print(f"\n--- [DCF 기반 분석 파라미터] ---")
    print(f"■ 로봇 이상 속도(SF): {SF} km/h")
    print(f"■ 도로 우회 계수(CF): {CF}")
    print(f"■ 보행/신호 지연 계수(DF): {DF}")
    print(f"■ 최종 변환 계수(DCF): {DCF:.4f} hours/km")
    print(f"■ 목표 제한 시간: {target_time_min}분")
    print(f"■ 필터링 기준 최대 직선 거리: {max_distance_km*1000:.1f}m ({max_distance_km:.3f} km)")
    print(f"---------------------------------\n")
    # =========================================================================
    
    # 위경도를 라디안으로 변환
    df_clean['lat_rad'] = np.radians(df_clean['lat'])
    df_clean['lon_rad'] = np.radians(df_clean['lon'])
    coords_rad = df_clean[['lat_rad', 'lon_rad']].values
    
    # 1. 하버사인 실제 거리 행렬 계산 (단위: km)
    dist_matrix = haversine_distances(coords_rad) * 6371.0088
    
    # 2. 계층적 군집화 (완전 연결법: Complete Linkage 적용)
    dist_array = squareform(dist_matrix)
    Z = linkage(dist_array, method='complete')
    
    # 3. DCF 산출 임계치 기준으로 군집 자르기
    df_clean['cluster'] = fcluster(Z, t=max_distance_km, criterion='distance')
    
    locations = df_clean[['가게명', 'lat', 'lon', 'cluster']].to_dict('records')
    locations_json = json.dumps(locations, ensure_ascii=False)
    
    spots = []
    unique_clusters = set(df_clean['cluster'])
    
    for cluster_id in unique_clusters:
        cluster_df = df_clean[df_clean['cluster'] == cluster_id]
        
        # 거점의 중심 좌표 (평균값)
        spot_lat = cluster_df['lat'].mean()
        spot_lon = cluster_df['lon'].mean()
        store_count = len(cluster_df)
        
        # 외곽 다각형 좌표 추출
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
        
    spots_json = json.dumps(spots, ensure_ascii=False)
    center_lat = df_clean['lat'].mean()
    center_lon = df_clean['lon'].mean()

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>네이버 지도 - DCF 기반 로봇 최적 거점 분석</title>
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
                content: '<div style="width:8px; height:8px; background:#007bef; border-radius:50%; border:1px solid white;"></div>',
                anchor: new naver.maps.Point(4, 4)
            }}
        }});
        var infoWindow = new naver.maps.InfoWindow({{
            content: '<div style="padding:10px;font-size:13px;font-weight:bold;">' + locations[i]['가게명'] + '</div>'
        }});
        (function(marker, infoWindow) {{
            naver.maps.Event.addListener(marker, 'click', function() {{
                if (infoWindow.getMap()) {{ infoWindow.close(); }} else {{ infoWindow.open(map, marker); }}
            }});
        }})(marker, infoWindow);
    }}

    // 2. DCF 기준 반경 보장 다각형 영역 및 거점 표시
    var spots = {spots_json};
    for (var j = 0; j < spots.length; j++) {{
        var hullData = spots[j].hull;
        var polygonPath = [];
        for (var k = 0; k < hullData.length; k++) {{
            polygonPath.push(new naver.maps.LatLng(hullData[k].lat, hullData[k].lon));
        }}

        if (polygonPath.length >= 3) {{
            var polygon = new naver.maps.Polygon({{
                map: map,
                paths: [polygonPath],
                fillColor: '#10b981', 
                fillOpacity: 0.15,
                strokeColor: '#059669',
                strokeOpacity: 0.6,
                strokeWeight: 2,
                clickable: false
            }});
        }} else if (polygonPath.length === 2) {{
            var polyline = new naver.maps.Polyline({{
                map: map,
                path: polygonPath,
                strokeColor: '#059669',
                strokeOpacity: 0.6,
                strokeWeight: 2
            }});
        }}

        // 거점 마커 생성
        var spotMarker = new naver.maps.Marker({{
            position: new naver.maps.LatLng(spots[j].lat, spots[j].lon),
            map: map,
            icon: {{
                content: '<div style="background:#10b981; color:white; padding:4px 8px; border-radius:12px; font-size:11px; font-weight:bold; border:2px solid white; box-shadow: 0px 2px 4px rgba(0,0,0,0.3); white-space:nowrap;">' +
                         '  거점 #' + (j+1) + ' (' + spots[j].count + '개점)' +
                         '</div>',
                anchor: new naver.maps.Point(40, 12)
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

    print(f"🎯 [성공] DCF 기반 10분 보장 거점 군집화 완료: {output_file}")
    print(f"생성된 최적의 배송 거점 개수: {len(spots)}개")

    # [평가 파트] (정상적인 위치로 이동 및 들여쓰기 교정 완료)
    # 1. 코페네틱 상관계수 (CPCC) 분석
    print("\n=======================================================")
    print("📊 [평가 1] 코페네틱 상관계수 (CPCC) 분석")
    print("=======================================================")
    
    c, coph_dists = cophenet(Z, dist_array)
    
    print(f"■ 산출된 코페네틱 상관계수: {c:.4f}")
    if c >= 0.8:
        print("■ 평가: 🌟 매우 우수 (원본 거리 데이터를 훌륭하게 보존하며 군집화됨)")
    elif c >= 0.6:
        print("■ 평가: 🟢 보통 (거리 데이터가 어느 정도 보존됨)")
    else:
        print("■ 평가: ⚠️ 낮음 (완전 연결법 적용 과정에서 거리 왜곡이 발생했을 수 있음)")

    # 2. 물리적 거리 제약 실효성 검증
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
    
    if violation_count == 0:
        print("■ 평가: 🎯 완벽함! 모든 거점(군집)이 10분 배송 제약 거리를 100% 만족합니다.")
    else:
        print("■ 평가: ❌ 주의! 제약 거리를 초과한 군집이 존재합니다. (임계치 조정 필요)")
    print("=======================================================\n")

# 최종 예외 처리 (데이터가 없는 경우)
else:
    print("❌ 좌표 변환된 데이터가 없습니다.")

#python -m http.server 8000
#http://localhost:8000/naver_store_map.html