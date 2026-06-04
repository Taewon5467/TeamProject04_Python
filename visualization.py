import json
import html as html_lib
import os
import threading
import webbrowser
import http.server
import socketserver
from config import NAVER_CLIENT_ID, OUTPUT_HTML


def generate_html(df_clean, spots, max_dist_km):
    locations_raw = df_clean[['가게명','lat','lon','cluster']].to_dict('records')
    locations_safe = [
        {k: html_lib.escape(str(v)) if isinstance(v, str) else v for k, v in loc.items()}
        for loc in locations_raw
    ]
    locations_json     = json.dumps(locations_safe, ensure_ascii=True)
    spots_json         = json.dumps(spots,          ensure_ascii=True)
    coverage_radius_m  = round(max_dist_km * 1000, 1)   # km → m 변환
    center_lat         = float(df_clean['lat'].mean())
    center_lon         = float(df_clean['lon'].mean())

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>로봇 최적 거점 분석 (ML 예측 모드)</title>
    <script type="text/javascript"
        src="https://oapi.map.naver.com/openapi/v3/maps.js?ncpKeyId={NAVER_CLIENT_ID}">
    </script>
    <style>
        html, body {{ margin:0; padding:0; width:100%; height:100%; }}
        #map {{ width:100%; height:100vh; }}
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
    locations.forEach(function(loc) {{
        new naver.maps.Marker({{
            position: new naver.maps.LatLng(loc.lat, loc.lon),
            map: map,
            icon: {{
                content: '<div style="width:6px;height:6px;background:#007bef;border-radius:50%;border:1px solid white;"></div>',
                anchor: new naver.maps.Point(3, 3)
            }}
        }});
    }});

    var spots = {spots_json};
    var coverageRadius = {coverage_radius_m};

    spots.forEach(function(spot, j) {{
        var polyPath = spot.hull.map(function(h) {{
            return new naver.maps.LatLng(h.lat, h.lon);
        }});
        if (polyPath.length >= 3) {{
            new naver.maps.Polygon({{
                map: map, paths: [polyPath],
                fillColor: '#10b981', fillOpacity: 0.15,
                strokeColor: '#059669', strokeOpacity: 0.6, strokeWeight: 2
            }});
        }}

        // ── 커버리지 원 ──────────────────────────────
        new naver.maps.Circle({{
            map: map,
            center: new naver.maps.LatLng(spot.lat, spot.lon),
            radius: coverageRadius,
            fillColor: '#3b82f6',
            fillOpacity: 0.08,
            strokeColor: '#2563eb',
            strokeOpacity: 0.5,
            strokeWeight: 1.5,
            strokeStyle: 'shortdash'
        }});

        new naver.maps.Marker({{
            position: new naver.maps.LatLng(spot.lat, spot.lon),
            map: map,
            icon: {{
                content:
                    '<div style="background:#0f172a;color:white;padding:5px 8px;border-radius:6px;' +
                    'font-family:sans-serif;border:1px solid #334155;box-shadow:0 4px 6px rgba(0,0,0,0.4);' +
                    'white-space:nowrap;text-align:center;line-height:1.4;">' +
                    '<div style="font-size:11px;font-weight:bold;color:#38bdf8;margin-bottom:2px;">' +
                    '거점 #' + (j+1) + ' (' + spot.count + '개 소속)</div>' +
                    '<div style="font-size:10px;color:#fde047;font-weight:bold;">' +
                    '🤖 피크 ' + spot.robots_peak + '대 | 평시 ' + spot.robots_avg + '대</div></div>',
                anchor: new naver.maps.Point(40, 15)
            }}
        }});
    }});
</script>
</body>
</html>"""

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"🗺️  HTML 생성 완료: {OUTPUT_HTML}")


# ── 서버 자동 시작 ────────────────────────────────────────
_server_started = False  # 중복 실행 방지

def open_map_in_browser(port=8000):
    """
    HTML이 있는 디렉토리 기준으로 HTTP 서버를 백그라운드 스레드로 실행,
    브라우저를 자동으로 열어줌. 이미 서버가 실행 중이면 브라우저만 열음.
    """
    global _server_started

    html_dir = os.path.dirname(os.path.abspath(OUTPUT_HTML))
    url      = f"http://localhost:{port}/naver_store_map.html"

    if not _server_started:
        def _serve():
            os.chdir(html_dir)
            handler = http.server.SimpleHTTPRequestHandler
            # 로그 출력 억제
            handler.log_message = lambda *args: None
            with socketserver.TCPServer(("", port), handler) as httpd:
                httpd.serve_forever()

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        _server_started = True
        print(f"🌐 로컬 서버 시작: http://localhost:{port}")

    # 서버가 뜰 때까지 잠깐 대기 후 브라우저 오픈
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"🔗 브라우저 자동 오픈: {url}")