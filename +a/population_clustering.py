import os  # 파일 존재 여부 확인용 추가
import pandas as pd
import requests
import json
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull, QhullError
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
 
# 1. 파일 경로 및 설정 정의
raw_file = "안산시_상록구_공동주택_기본정보.csv"
cached_file = "안산시_상록구_공동주택_좌표포함.csv"  # 좌표가 저장될 캐시 파일

NAVER_CLIENT_ID = 'f2654c4us6'
NAVER_CLIENT_SECRET = 'jMvT8jiXVaDPhyDwuIWZWGoxj2XCIGyrVWzv3ZWn'

# 2. 데이터 로드 및 좌표 변환 (캐싱 메커니즘 적용)
if os.path.exists(cached_file):
    print(f"이미 변환된 좌표 파일({cached_file})을 찾았습니다. API 호출을 건너뛰고 고속 로드합니다.")
    try:
        df_clean = pd.read_csv(cached_file, encoding='cp949')
    except UnicodeDecodeError:
        df_clean = pd.read_csv(cached_file, encoding='utf-8')
    print(f"➔ 로드 완료! (성공 데이터: {len(df_clean)}개)")

else:
    print(f"캐시 파일이 없습니다. 최초 1회 네이버 API 좌표 변환을 시작합니다.")
    
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
    
    df_clean.to_csv(cached_file, index=False, encoding='cp949')
    print(f"좌표 변환 데이터가 {cached_file} 파일로 영구 보관되었습니다.")

# 주소 필드에서 행정동(예: 사동, 본오동, 성포동 등)을 안전하게 추출하여 '동명' 컬럼 생성
# 주소의 3번째 단어(안산시 상록구 'XX동')를 타겟팅합니다.
df_clean['동명'] = df_clean['주소'].astype(str).apply(lambda x: x.split()[2] if len(x.split()) > 2 else "기타지역")

# ==========================================
# 3. ⚖️ 행정동별 독립 가중치 보정 및 K-means 군집화 (MinMaxScaler 패키지 연동형)
# ==========================================
if not df_clean.empty:
    unique_dongs = df_clean['동명'].unique()
    
    all_locations = []
    all_spots = []
    global_cluster_counter = 0  
    total_calculated_stations = 0 
    
    print(f"\n 상록구 내 총 {len(unique_dongs)}개 행정동 독립 최적 군집화 시동...")

    for dong in unique_dongs:
        # 1) 해당 행정동의 데이터만 격리
        dong_df = df_clean[df_clean['동명'] == dong].copy()
        
        if len(dong_df) < 3:
            print(f"  - [{dong:5s}] 데이터 부족({len(dong_df)}개) ➔ 1개 기본 거점으로 지정")
            dong_df['global_cluster'] = global_cluster_counter
            all_locations.extend(dong_df[['단지명_공시가격', 'lat', 'lon', 'global_cluster', '세대수']].to_dict('records'))
            
            total_households = int(dong_df['세대수'].sum())
            all_spots.append({
                "cluster_id": global_cluster_counter,
                "lat": dong_df['lat'].mean(),
                "lon": dong_df['lon'].mean(),
                "apt_count": len(dong_df),
                "total_households": total_households,
                "hull": [{"lat": float(p['lat']), "lon": float(p['lon'])} for _, p in dong_df.iterrows()],
                "dong_name": dong
            })
            global_cluster_counter += 1
            total_calculated_stations += 1
            continue

        # 행정동 내부 거리 분포 기반 고립지 필터링
        neighbors = NearestNeighbors(n_neighbors=2, metric='euclidean')
        neighbors_fit = neighbors.fit(dong_df[['lat', 'lon']])
        distances, _ = neighbors_fit.kneighbors(dong_df[['lat', 'lon']])
        sorted_distances = np.sort(distances[:, 1])
        
        optimal_eps = np.percentile(sorted_distances, 90.0)
        optimal_eps = max(optimal_eps, 0.0015) 
        
        spatial_filter = DBSCAN(eps=optimal_eps, min_samples=2, metric='euclidean')
        spatial_labels = spatial_filter.fit_predict(dong_df[['lat', 'lon']])
        
        before_dong_len = len(dong_df)
        dong_df = dong_df[spatial_labels != -1].copy()
        
        if len(dong_df) < 2:
            dong_df = df_clean[df_clean['동명'] == dong].copy()
            
        # 2) 세대수 가중치 스케일링 변환 완료
        raw_weights = dong_df['세대수'].values
        min_gen, max_gen = np.min(raw_weights), np.max(raw_weights)
        
        if max_gen != min_gen:
            # 1. 1차원 데이터 배열을 2차원 매트릭스 형태로 정렬
            raw_weights_2d = raw_weights.reshape(-1, 1)
            
            # 2.가중치 하한선(0.15)과 상한선(1.0) 파라미터 적용
            scaler = MinMaxScaler(feature_range=(0.15, 1.0))
            
            # 3. 변환 연산 수행 후, KMeans 가중치 규격에 맞춰 1차원 배열로 평탄화
            balanced_weights = scaler.fit_transform(raw_weights_2d).flatten()
        else:
            # 최댓값과 최솟값이 같은 동네는 스케일러 연산 생략 후 하한 균등 배분
            balanced_weights = np.ones(len(dong_df)) * 0.5

        # 동별 독립 가중 실루엣 점수 스캔 (K 최적화 탐색)
        best_k = 2
        best_score = -1
        max_k_to_test = min(9, len(dong_df)) 
        
        if max_k_to_test > 2:
            for k in range(2, max_k_to_test):
                test_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                test_labels = test_kmeans.fit_predict(dong_df[['lat', 'lon']], sample_weight=balanced_weights)
                
                sample_scores = silhouette_samples(dong_df[['lat', 'lon']], test_labels)
                weighted_score = np.sum(sample_scores * balanced_weights) / np.sum(balanced_weights)
                
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_k = k
        else:
            best_k = 2
                
        print(f"  - [{dong:5s}] ➔ 수학적 최적 스테이션 개수: [{best_k}]개 확정 (실루엣 점수: {best_score:.4f} / 잔존단지: {len(dong_df)}개)")
        total_calculated_stations += best_k

        # 확정된 최적 K로 해당 동 최종 가중 K-means 학습
        final_clusterer = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        dong_df['local_cluster'] = final_clusterer.fit_predict(dong_df[['lat', 'lon']], sample_weight=balanced_weights)
        
        dong_df['global_cluster'] = dong_df['local_cluster'] + global_cluster_counter
        all_locations.extend(dong_df[['단지명_공시가격', 'lat', 'lon', 'global_cluster', '세대수']].to_dict('records'))

        # 군집별 최종 가중치 중심점 및 Convex Hull 추출
        for local_c_id in range(best_k):
            cluster_df = dong_df[dong_df['local_cluster'] == local_c_id]
            if cluster_df.empty: continue
            
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
                
            all_spots.append({
                "cluster_id": global_cluster_counter + local_c_id,
                "lat": spot_lat,
                "lon": spot_lon,
                "apt_count": apt_count,
                "total_households": total_households,
                "hull": hull_coordinates,
                "dong_name": dong
            })
            
        global_cluster_counter += best_k

    # JavaScript 연동을 위해 JSON 직렬화 변환
    locations_json = json.dumps(all_locations, ensure_ascii=False)
    spots_json = json.dumps(all_spots, ensure_ascii=False)
    center_lat = df_clean['lat'].mean()
    center_lon = df_clean['lon'].mean()
        
    # ==========================================
    # 4.  네이버 지도가 포함된 HTML 파일 빌드
    # ==========================================
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>네이버 지도 - 행정동별 최적 K-Means 가중 군집화</title>
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
    // 시각적 겹침 방지용 글로벌 컬러 팔레트 
    var clusterColors = [
        '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#06B6D4', '#78350F', 
        '#4B5563', '#1E40AF', '#065F46', '#92400E', '#991B1B', '#3730A3', '#D946EF', '#14B8A6'
    ];
    
    // 1. 단지별 점 마커 찍기
    for (var i = 0; i < locations.length; i++) {{
        var cId = locations[i].global_cluster;
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

    // 2. 행정동별 거점 및 배달 영역(Convex Hull) 그리기
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
                         '🚀 [' + spots[j].dong_name + '] 거점 (' + spots[j].total_households.toLocaleString() + '세대 담당)' +
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

    print(f"\n 가중치 보정 K-means 분석 및 네이버 지도 빌드 완료: {output_html}")
    print(f" 상록구 전체 행정동에 배치된 최종 로봇 거점 총합: {total_calculated_stations}개")