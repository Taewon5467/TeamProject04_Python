import pandas as pd
import requests
import json
from sklearn.cluster import DBSCAN  
from scipy.spatial import ConvexHull, QhullError
import numpy as np

file_name = r'C:\MlProject\TeamProject04_Python\상록구 일동_통합_가게정보.csv'

try:
    df = pd.read_csv(file_name, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_name, encoding='cp949')

if '주소' not in df.columns or '가게명' not in df.columns:
    print("❌ CSV 파일에 '주소' 또는 '가게명' 컬럼이 없습니다. 컬럼명을 확인해 주세요.")
    print(f"현재 컬럼명: {list(df.columns)}")
    exit()

# 주소 (몇 층 등 상세 주소 제거)
df['정제된_주소'] = df['주소'].astype(str).str.split(' ').str[:5].str.join(' ')

# 본인의 API 키
NAVER_CLIENT_ID = 'hymjc5hhjr'
NAVER_CLIENT_SECRET = 'GoSPNaa2QmBu4WPTw4bNOlWvYWgGk76sYshHodDX'

def get_naver_lat_lon(address):
    url = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET
    }
    params = {
        "query": address
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        res_json = response.json()
        
        if 'addresses' in res_json and res_json['addresses']:
            y = float(res_json['addresses'][0]['y']) # 위도(lat)
            x = float(res_json['addresses'][0]['x']) # 경도(lon)
            return pd.Series([y, x])
        else:
            print(f"❌ [에러/실패] 주소: '{address}' | 네이버 응답: {res_json}")
            
    except Exception as e:
        print(f"⚠️ 요청 중 파이썬 에러 발생: {e}")
        
    return pd.Series([None, None])

print("네이버 API로 주소를 좌표로 변환하고 있습니다...")
df[['lat', 'lon']] = df['정제된_주소'].apply(get_naver_lat_lon)


df_clean = df.dropna(subset=['lat', 'lon']).copy()

if not df_clean.empty:
    # eps = 0.002 는 180m  이내의 min_samples =3 은 최소 3개 이상 가게가 모여야 하나의 스팟으로 인정
    dbscan = DBSCAN(eps=0.0018, min_samples=3, metric='euclidean')
    df_clean['cluster'] = dbscan.fit_predict(df_clean[['lat', 'lon']])
    
    locations = df_clean[['가게명', 'lat', 'lon', 'cluster']].to_dict('records')
    locations_json = json.dumps(locations, ensure_ascii=False)
    
    spots = []
    unique_clusters = set(df_clean['cluster']) - {-1}  
    
    for cluster_id in unique_clusters:
        cluster_df = df_clean[df_clean['cluster'] == cluster_id]
        
        # 최적의 스팟 위치 = 해당 군집 데이터의 위도/경도 평균값
        spot_lat = cluster_df['lat'].mean()
        spot_lon = cluster_df['lon'].mean()
        store_count = len(cluster_df)
        
        # 외곽 라인 좌표를 구하기 위한 알고리즘 (Convex Hull) 적용
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
    # =========================================================================

    center_lat = df_clean['lat'].mean()
    center_lon = df_clean['lon'].mean()


    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>네이버 지도 - 최적 거점 분석</title>
    <script type="text/javascript" src="https://oapi.map.naver.com/openapi/v3/maps.js?ncpKeyId={NAVER_CLIENT_ID}"></script>
    <style>
        html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; }}
        #map {{ width: 100%; height: 100vh; }}
    </style>
</head>
<body>

<div id="map"></div>

<script>
    // 지도 생성
    var map = new naver.maps.Map('map', {{
        center: new naver.maps.LatLng({center_lat}, {center_lon}),
        zoom: 14
    }});

    // 일반 가게 데이터 마커 표시
    var locations = {locations_json};
    for (var i = 0; i < locations.length; i++) {{
        // 군집에 속하지 못한 외딴 가게(-1)는 회색, 군집된 가게는 파란색 소형 마커 처리
        var markerColor = locations[i].cluster === -1 ? '#aaaaaa' : '#007bef';
        
        var marker = new naver.maps.Marker({{
            position: new naver.maps.LatLng(locations[i].lat, locations[i].lon),
            map: map,
            icon: {{
                content: '<div style="width:8px; height:8px; background:' + markerColor + '; border-radius:50%; border:1px solid white;"></div>',
                anchor: new naver.maps.Point(4, 4)
            }}
        }});

        // 정보창 붙이기
        var infoWindow = new naver.maps.InfoWindow({{
            content: '<div style="padding:10px;font-size:13px;font-weight:bold;">' + locations[i]['가게명'] + '</div>'
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

    // 군집의 외곽 라인(Polygon)을 따라 시각화
    var spots = {spots_json};
    for (var j = 0; j < spots.length; j++) {{
        var hullData = spots[j].hull;
        var polygonPath = [];
        
        // 네이버 지도 API 전용 LatLng 좌표 배열 생성
        for (var k = 0; k < hullData.length; k++) {{
            polygonPath.push(new naver.maps.LatLng(hullData[k].lat, hullData[k].lon));
        }}

        // 외곽선 면적 그리기
        if (polygonPath.length >= 3) {{
            var polygon = new naver.maps.Polygon({{
                map: map,
                paths: [polygonPath],
                fillColor: '#ff3333',
                fillOpacity: 0.2,       // 다각형 내부 투명도
                strokeColor: '#ff0000',   // 라인 색상
                strokeOpacity: 0.7,     // 라인 투명도
                strokeWeight: 2,         // 라인 두께
                clickable: false
            }});
        }} else if (polygonPath.length === 2) {{
           
        
            var polyline = new naver.maps.Polyline({{
                map: map,
                path: polygonPath,
                strokeColor: '#ff0000',
                strokeOpacity: 0.7,
                strokeWeight: 2
            }});
        }}

        // 최적 스팟 중심에 텍스트형 디자인 마커 생성
        var spotMarker = new naver.maps.Marker({{
            position: new naver.maps.LatLng(spots[j].lat, spots[j].lon),
            map: map,
            icon: {{
                content: '<div style="background:#ef4444; color:white; padding:4px 8px; border-radius:12px; font-size:11px; font-weight:bold; border:2px solid white; box-shadow: 0px 2px 4px rgba(0,0,0,0.3); white-space:nowrap;">' +
                         ' 거점 #' + (j+1) + ' (' + spots[j].count + '개점)' +
                         '</div>',
                anchor: new naver.maps.Point(40, 12)
            }}
        }});
    }}
</script>

</body>
</html>
"""

    # HTML 저장
    output_file = 'naver_store_map.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"알고리즘 분석 및 영역 지도 생성 완료: {output_file}")
    print(f"발견된 최적의 거점(스팟) 개수: {len(spots)}개")

else:
    print(" 좌표 변환된 데이터가 없습니다.")