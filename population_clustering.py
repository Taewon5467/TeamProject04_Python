import os  # 파일 존재 여부 확인용 추가
import pandas as pd
import requests
import json
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, QhullError
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
# python -m http.server 8000
# 
# ==========================================
# 1. 파일 경로 및 설정 정의
# ==========================================
raw_file = "안산시_상록구_공동주택_기본정보.csv"
cached_file = "안산시_상록구_공동주택_좌표포함.csv"  # 💾 좌표가 저장될 캐시 파일

NAVER_CLIENT_ID = 'f2654c4us6'
NAVER_CLIENT_SECRET = 'jMvT8jiXVaDPhyDwuIWZWGoxj2XCIGyrVWzv3ZWn'

# ==========================================
# 2. 데이터 로드 및 좌표 변환 (캐싱 메커니즘 적용)
# ==========================================

# [조건 검사] 이미 과거에 좌표 변환을 완료한 파일이 로컬에 존재하는가?
if os.path.exists(cached_file):
    print(f"✨ 이미 변환된 좌표 파일({cached_file})을 찾았습니다. API 호출을 건너뛰고 고속 로드합니다.")
    try:
        df_clean = pd.read_csv(cached_file, encoding='cp949')
    except UnicodeDecodeError:
        df_clean = pd.read_csv(cached_file, encoding='utf-8')
    print(f"➔ 로드 완료! (성공 데이터: {len(df_clean)}개)")

else:
    print(f"🚨 캐시 파일이 없습니다. 최초 1회 네이버 API 좌표 변환을 시작합니다.")
    
    try:
        df = pd.read_csv(raw_file, encoding='cp949')
    except UnicodeDecodeError:
        df = pd.read_csv(raw_file, encoding='utf-8')

    df['정제된_주소'] = df['주소'].astype(str).apply(lambda x: " ".join(x.split()[:5]))

    def fetch_lat_lon(idx, address):
        url = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
            "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET
        }
        params = {"query": address}
        try:
            response = requests.get(url, headers=headers, params=params, timeout=5)
            res_json = response.json()
            if 'addresses' in res_json and res_json['addresses']:
                y = float(res_json['addresses'][0]['y']) # 위도
                x = float(res_json['addresses'][0]['x']) # 경도
                return idx, y, x
        except Exception as e:
            pass
        return idx, None, None

    df['lat'] = None
    df['lon'] = None

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = [
            executor.submit(fetch_lat_lon, idx, row['정제된_주소']) 
            for idx, row in df.iterrows()
        ]
        for i, future in enumerate(as_completed(futures), 1):
            idx, lat, lon = future.result()
            df.at[idx, 'lat'] = lat
            df.at[idx, 'lon'] = lon
            if i % 50 == 0 or i == len(df):
                print(f"   - 좌표 변환 진행률: {i}/{len(df)} 완료...")

    # 정제 및 타입 변환
    df_clean = df.dropna(subset=['lat', 'lon', '세대수']).copy()
    df_clean['lat'] = df_clean['lat'].astype(float)
    df_clean['lon'] = df_clean['lon'].astype(float)
    df_clean['세대수'] = df_clean['세대수'].astype(int)
    
    # 💾 중요: 다음 실행 때 써먹을 수 있도록 변환 결과를 로컬에 저장!
    df_clean.to_csv(cached_file, index=False, encoding='cp949')
    print(f"💾 좌표 변환 데이터가 {cached_file} 파일로 영구 보관되었습니다.")

# ==========================================
# 3. ⚖️ K-means 군집화 알고리즘 실행 (여기부터는 매번 즉시 실행됨)
# ==========================================
if not df_clean.empty:
    # 하이퍼파라미터 조정을 자유롭게 테스트하세요!
    K = 20  
    min_w, max_w = 0.15, 1.0
    
    raw_weights = df_clean['세대수'].values
    min_gen, max_gen = np.min(raw_weights), np.max(raw_weights)
    
    norm_weights = (raw_weights - min_gen) / (max_gen - min_gen + 1e-8)
    balanced_weights = norm_weights * (max_w - min_w) + min_w

    clusterer = KMeans(n_clusters=K, random_state=42, n_init=10)
    df_clean['cluster'] = clusterer.fit_predict(df_clean[['lat', 'lon']], sample_weight=balanced_weights)
    
    locations = df_clean[['단지명_공시가격', 'lat', 'lon', 'cluster', '세대수']].to_dict('records')
    locations_json = json.dumps(locations, ensure_ascii=False)
    
    spots = []
    unique_clusters = set(df_clean['cluster'])
    
    for cluster_id in unique_clusters:
        cluster_df = df_clean[df_clean['cluster'] == cluster_id]
        
        total_households = int(cluster_df['세대수'].sum())
        spot_lat = (cluster_df['lat'] * cluster_df['세대수']).sum() / total_households
        spot_lon = (cluster_df['lon'] * cluster_df['세대수']).sum() / total_households
        apt_count = len(cluster_df)
        
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
            "apt_count": apt_count,
            "total_households": total_households,
            "hull": hull_coordinates
        })
        
    spots_json = json.dumps(spots, ensure_ascii=False)
    center_lat = df_clean['lat'].mean()
    center_lon = df_clean['lon'].mean()
        
    # ==========================================
    # 4. 🗺️ 네이버 지도가 포함된 HTML 파일 빌드
    # ==========================================
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>네이버 지도 - 아파트 K-Means 가중치 균형 군집화</title>
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

    var locations = {locations_json};
    var clusterColors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#06B6D4', '#78350F', '#4B5563'];
    
    for (var i = 0; i < locations.length; i++) {{
        var cId = locations[i].cluster;
        var markerColor = clusterColors[cId % clusterColors.length];
        var markerSize = locations[i].세대수 > 1500 ? 15 : (locations[i].세대수 > 700 ? 10 : 6);
        
        var marker = new naver.maps.Marker({{
            position: new naver.maps.LatLng(locations[i].lat, locations[i].lon),
            map: map,
            icon: {{
                content: '<div style="width:'+markerSize+'px; height:'+markerSize+'px; background:' + markerColor + '; border-radius:50%; border:1.5px solid white; box-shadow: 0 1px 4px rgba(0,0,0,0.4);"></div>',
                anchor: new naver.maps.Point(markerSize/2, markerSize/2)
            }}
        }});

        var infoWindow = new naver.maps.InfoWindow({{
            content: '<div style="padding:10px;font-size:12px;font-family:sans-serif;"><b>' + locations[i]['단지명_공시가격'] + '</b><br>수요 규모: ' + locations[i].세대수.toLocaleString() + '세대</div>'
        }});

        (function(marker, infoWindow) {{
            naver.maps.Event.addListener(marker, 'click', function() {{
                if (infoWindow.getMap()) {{
                    infoWindow.close();
                }} else {{
                    infoWindow.open(map, marker);
                }}
            }});
        }})(marker, infoWindow);
    }}

    var spots = {spots_json};
    for (var j = 0; j < spots.length; j++) {{
        var hullData = spots[j].hull;
        var polygonPath = [];
        var currentClusterColor = clusterColors[spots[j].cluster_id % clusterColors.length];
        
        for (var k = 0; k < hullData.length; k++) {{
            polygonPath.push(new naver.maps.LatLng(hullData[k].lat, hullData[k].lon));
        }}

        if (polygonPath.length >= 3) {{
            var polygon = new naver.maps.Polygon({{
                map: map,
                paths: [polygonPath],
                fillColor: currentClusterColor,     
                fillOpacity: 0.12,       
                strokeColor: currentClusterColor,   
                strokeOpacity: 0.7,     
                strokeWeight: 2.5,         
                clickable: false
            }});
        }}

        var spotMarker = new naver.maps.Marker({{
            position: new naver.maps.LatLng(spots[j].lat, spots[j].lon),
            map: map,
            icon: {{
                content: '<div style="background:' + currentClusterColor + '; color:white; padding:6px 14px; border-radius:30px; font-size:11px; font-weight:bold; border:2.5px solid white; box-shadow: 0px 3px 6px rgba(0,0,0,0.4); white-space:nowrap; font-family:sans-serif;">' +
                         '🚀 최적 거점 #' + (spots[j].cluster_id + 1) + ' (' + spots[j].total_households.toLocaleString() + '세대 담당)' +
                         '</div>',
                anchor: new naver.maps.Point(60, 15)
            }}
        }});
    }}
</script>
</body>
</html>
"""
    output_html = 'naver_apartment_kmeans_map.html'
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    df_clean.to_csv('안산시_상록구_아파트_KMeans_결과.csv', index=False, encoding='cp949')

    print(f"\n✅ 가중치 보정 K-means 분석 및 네이버 지도 빌드 완료: {output_html}")
    print(f"📌 배치된 배달 로봇 최적 스테이션 거점 개수: {K}개")