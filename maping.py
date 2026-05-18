import pandas as pd
import requests
import json

file_name = r'C:\MlProject\local\상록구 일동_통합_가게정보.csv'

try:
    df = pd.read_csv(file_name, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_name, encoding='cp949')


if '주소' not in df.columns or '가게명' not in df.columns:
    print("❌ CSV 파일에 '주소' 또는 '가게명' 컬럼이 없습니다. 컬럼명을 확인해 주세요.")
    print(f"현재 컬럼명: {list(df.columns)}")
    exit()

# 3. 주소 정제 (몇 층 등 상세 주소 제거)
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
        # params를 사용해 안전하게 URL을 요청합니다.
        response = requests.get(url, headers=headers, params=params)
        res_json = response.json()
        
        # 정상적으로 주소를 찾은 경우
        if 'addresses' in res_json and res_json['addresses']:
            y = float(res_json['addresses'][0]['y']) # 위도(lat)
            x = float(res_json['addresses'][0]['x']) # 경도(lon)
            return pd.Series([y, x])
        else:
            # 주소를 못 찾았거나 에러가 발생한 경우 이유를 출력합니다.
            print(f"❌ [에러/실패] 주소: '{address}' | 네이버 응답: {res_json}")
            
    except Exception as e:
        print(f"⚠️ 요청 중 파이썬 에러 발생: {e}")
        
    return pd.Series([None, None])

print("네이버 API로 주소를 좌표로 변환하고 있습니다...")
df[['lat', 'lon']] = df['정제된_주소'].apply(get_naver_lat_lon)

# 좌표를 찾은 정상 데이터만 남기기
df_clean = df.dropna(subset=['lat', 'lon'])

if not df_clean.empty:
    center_lat = df_clean['lat'].mean()
    center_lon = df_clean['lon'].mean()
    
    # 파이썬 데이터를 자바스크립트가 읽을 수 있는 JSON 형태로 변환
    locations = df_clean[['가게명', 'lat', 'lon']].to_dict('records')
    locations_json = json.dumps(locations, ensure_ascii=False)

   # HTML 생성
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>네이버 지도</title>

    <script
    <script type="text/javascript" src="https://oapi.map.naver.com/openapi/v3/maps.js?ncpKeyId={NAVER_CLIENT_ID}"></script>
    </script>

    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
        }}

        #map {{
            width: 100%;
            height: 100vh;
        }}
    </style>
</head>

<body>

<div id="map"></div>

<script>

    // 지도 생성
    var map = new naver.maps.Map('map', {{
        center: new naver.maps.LatLng({center_lat}, {center_lon}),
        zoom: 15
    }});

    // 가게 데이터
    var locations = {locations_json};

    // 마커 생성
    for (var i = 0; i < locations.length; i++) {{

        var marker = new naver.maps.Marker({{
            position: new naver.maps.LatLng(
                locations[i].lat,
                locations[i].lon
            ),
            map: map
        }});

        // 정보창
        var infoWindow = new naver.maps.InfoWindow({{
            content:
                '<div style="padding:10px;font-size:14px;font-weight:bold;">'
                + locations[i]['가게명'] +
                '</div>'
        }});

        // 클릭 이벤트
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

</script>

</body>
</html>
"""

    # HTML 저장
    output_file = 'naver_store_map.html'

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"✅ 지도 생성 완료: {output_file}")

else:
    print("❌ 좌표 변환된 데이터가 없습니다.")