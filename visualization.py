import json
import html as html_lib
import os
import threading
import webbrowser
import http.server
import socketserver
import time
import sys
from config import NAVER_CLIENT_ID, OUTPUT_HTML


def generate_html(df_clean, spots, max_dist_km):
    import re

    # ── 동 정보 추출 ──────────────────────────────────────────
    def extract_dong(addr):
        m = re.search(r'(\S+동)', str(addr))
        return m.group(1) if m else '기타'

    df_work = df_clean.copy()

    if '주소' not in df_work.columns:
        print("⚠️  '주소' 컬럼이 없습니다. 동 색상 구분을 사용할 수 없습니다.")
        df_work['dong'] = '기타'
    else:
        df_work['dong'] = df_work['주소'].apply(extract_dong)

    dong_counts = df_work['dong'].value_counts().to_dict()
    print(f"🏘️  동별 가게 수: {dong_counts}")

    # ── 색상 팔레트 ───────────────────────────────────────────
    dong_list = sorted(df_work['dong'].unique().tolist())
    palette = [
        '#FF0000',  
        '#0055FF',  
        '#00CC00',  
        '#FF9900',  
        '#CC00CC',  
        '#00CCCC',  
        '#FFEE00',  
        '#FF0099',  
        '#6600FF',  
        '#00FF66',  
    ]
    dong_color_map = {dong: palette[i % len(palette)] for i, dong in enumerate(dong_list)}

    # ── 거점별 주요 동 할당 (소속 가게 중 최다 동) ────────────
    spots = [dict(s) for s in spots]
    for spot in spots:
        cid = spot['cluster_id']
        cluster_df = df_work[df_work['cluster'] == cid]
        if not cluster_df.empty and 'dong' in cluster_df.columns:
            dominant_dong = cluster_df['dong'].mode()[0]
        else:
            dominant_dong = '기타'
        spot['dong'] = dominant_dong

    # ── JSON 직렬화 ───────────────────────────────────────────
    locations_raw = df_work[['가게명', 'lat', 'lon', 'cluster', 'dong']].to_dict('records')
    locations_safe = [
        {k: html_lib.escape(str(v)) if isinstance(v, str) else v for k, v in loc.items()}
        for loc in locations_raw
    ]

    locations_json    = json.dumps(locations_safe, ensure_ascii=True)
    dong_color_json   = json.dumps(dong_color_map, ensure_ascii=True)
    dong_list_json    = json.dumps(dong_list,       ensure_ascii=True)
    spots_json        = json.dumps(spots,           ensure_ascii=True)
    coverage_radius_m = round(max_dist_km * 1000, 1)
    center_lat        = float(df_clean['lat'].mean())
    center_lon        = float(df_clean['lon'].mean())

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

        /* ── 범례 패널 ── */
        #legend {{
            position: absolute;
            bottom: 30px;
            right: 10px;
            background: rgba(15,23,42,0.92);
            color: white;
            padding: 14px 16px;
            border-radius: 10px;
            font-family: sans-serif;
            font-size: 12px;
            border: 1px solid #334155;
            box-shadow: 0 4px 16px rgba(0,0,0,0.5);
            z-index: 1000;
            min-width: 210px;
            user-select: none;
        }}
        #legend .legend-title {{
            font-size: 16px;
            font-weight: bold;
            color: #94a3b8;
            margin-bottom: 8px;
            letter-spacing: 0.05em;
        }}
        #legend .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 6px;
            gap: 8px;
            cursor: pointer;
        }}
        #legend .legend-item:hover {{
            opacity: 0.8;
        }}
        #legend .legend-dot {{
            width: 13px;
            height: 13px;
            border-radius: 50%;
            border: 1.5px solid rgba(255,255,255,0.5);
            flex-shrink: 0;
        }}
        #legend .legend-label {{
            flex: 1;
            font-size: 16px;
        }}

        /* ── 토글 스위치 ── */
        .toggle-switch {{
            position: relative;
            width: 32px;
            height: 17px;
            flex-shrink: 0;
        }}
        .toggle-switch input {{
            opacity: 0;
            width: 0;
            height: 0;
            position: absolute;
        }}
        .toggle-slider {{
            position: absolute;
            inset: 0;
            background: #475569;
            border-radius: 17px;
            transition: background 0.2s;
            cursor: pointer;
        }}
        .toggle-slider::before {{
            content: '';
            position: absolute;
            width: 11px;
            height: 11px;
            left: 3px;
            top: 3px;
            background: white;
            border-radius: 50%;
            transition: transform 0.2s;
        }}
        .toggle-switch input:checked + .toggle-slider {{
            background: var(--dong-color, #3b82f6);
        }}
        .toggle-switch input:checked + .toggle-slider::before {{
            transform: translateX(15px);
        }}

        /* ── 전체 토글 버튼 ── */
        #toggle-all-btn {{
            margin-top: 8px;
            width: 100%;
            padding: 5px 0;
            background: #1e293b;
            color: #94a3b8;
            border: 1px solid #334155;
            border-radius: 6px;
            font-size: 11px;
            cursor: pointer;
            text-align: center;
        }}
        #toggle-all-btn:hover {{
            background: #334155;
            color: white;
        }}
    </style>
</head>
<body>
<div id="map"></div>
<div id="legend">
    <div class="legend-title">📍 동별 커버리지 원</div>
    <div id="legend-items"></div>
    <button id="toggle-all-btn">전체 끄기</button>
</div>
<script>
    var map = new naver.maps.Map('map', {{
        center: new naver.maps.LatLng({center_lat}, {center_lon}),
        zoom: 14
    }});

    var dongColorMap = {dong_color_json};
    var dongList     = {dong_list_json};

    // ── 동별 원 객체 저장소 ─────────────────────────────────
    // dongCircles[동이름] = [Circle, Circle, ...]
    var dongCircles = {{}};
    dongList.forEach(function(d) {{ dongCircles[d] = []; }});

    // ── 가게 마커 ───────────────────────────────────────────
    var locations = {locations_json};
    locations.forEach(function(loc) {{
        var color = dongColorMap[loc.dong] || '#888888';
        new naver.maps.Marker({{
            position: new naver.maps.LatLng(loc.lat, loc.lon),
            map: map,
            icon: {{
                content: '<div style="width:7px;height:7px;background:' + color +
                         ';border-radius:50%;border:1.5px solid white;' +
                         'box-shadow:0 1px 3px rgba(0,0,0,0.4);"></div>',
                anchor: new naver.maps.Point(3, 3)
            }}
        }});
    }});

    // ── 거점: 다각형 + 커버리지 원 + 라벨 ────────────────────
    var spots = {spots_json};
    var coverageRadius = {coverage_radius_m};
    var spotMarkers = [];  // 줌 레벨별 라벨 전환용 저장소
    var ZOOM_THRESHOLD = 16;  // 이 줌 이상일 때만 상세 라벨 표시

    function makeCompactIcon(spot, color, idx) {{
    return {{
        content:
            '<div style="background:#0f172a;color:' + color + ';' +
            'width:34px;height:34px;border-radius:50%;border:3px solid ' + color + ';' +
            'font-family:sans-serif;font-size:17px;font-weight:bold;' +
            'box-shadow:0 3px 8px rgba(0,0,0,0.6);' +
            'display:flex;align-items:center;justify-content:center;">' +
            (idx + 1) + '</div>',
        anchor: new naver.maps.Point(17, 17)
    }};
}}

    function makeFullIcon(spot, color, idx) {{
    return {{
        content:
            '<div style="background:#0f172a;color:white;padding:12px 16px;border-radius:8px;' +
            'font-family:sans-serif;border:3px solid ' + color + ';' +
            'box-shadow:0 5px 10px rgba(0,0,0,0.5);white-space:nowrap;text-align:center;line-height:1.7;">' +
            '<div style="font-size:22px;font-weight:bold;color:' + color + ';margin-bottom:5px;">' +
            '거점 #' + (idx + 1) + ' (' + spot.count + '개 소속)</div>' +
            '<div style="font-size:19px;color:#fde047;font-weight:bold;">' +
            '🤖 피크 ' + spot.robots_peak + '대 | 평시 ' + spot.robots_avg + '대</div></div>',
        anchor: new naver.maps.Point(70, 26)
    }};
}}

    spots.forEach(function(spot, j) {{
        var baseColor = dongColorMap[spot.dong] || '#3b82f6';

        // 다각형
        var polyPath = spot.hull.map(function(h) {{
            return new naver.maps.LatLng(h.lat, h.lon);
        }});
        if (polyPath.length >= 3) {{
            new naver.maps.Polygon({{
                map: map,
                paths: [polyPath],
                fillColor:     baseColor,
                fillOpacity:   0.15,
                strokeColor:   baseColor,
                strokeOpacity: 0.7,
                strokeWeight:  2
            }});
        }}

        // 커버리지 원
        var circle = new naver.maps.Circle({{
            map: map,
            center: new naver.maps.LatLng(spot.lat, spot.lon),
            radius: coverageRadius,
            fillColor:     baseColor,
            fillOpacity:   0.07,
            strokeColor:   baseColor,
            strokeOpacity: 0.55,
            strokeWeight:  1.5,
            strokeStyle:   'shortdash'
        }});
        if (dongCircles[spot.dong]) {{
            dongCircles[spot.dong].push(circle);
        }}

        // 거점 라벨 (초기 줌 레벨에 맞춰 컴팩트/상세 아이콘 결정)
        var initialIcon = map.getZoom() >= ZOOM_THRESHOLD
            ? makeFullIcon(spot, baseColor, j)
            : makeCompactIcon(spot, baseColor, j);

        var labelMarker = new naver.maps.Marker({{
            position: new naver.maps.LatLng(spot.lat, spot.lon),
            map: map,
            icon: initialIcon
        }});
        spotMarkers.push({{ marker: labelMarker, spot: spot, color: baseColor, idx: j }});
    }});

    // 줌 변경 시 라벨 아이콘 전환
    naver.maps.Event.addListener(map, 'zoom_changed', function() {{
        var zoom = map.getZoom();
        spotMarkers.forEach(function(item) {{
            var icon = zoom >= ZOOM_THRESHOLD
                ? makeFullIcon(item.spot, item.color, item.idx)
                : makeCompactIcon(item.spot, item.color, item.idx);
            item.marker.setIcon(icon);
        }});
    }});

    // ── 범례 + 토글 렌더링 ───────────────────────────────────
    var legendEl   = document.getElementById('legend-items');
    var dongStates = {{}};   // 동별 on/off 상태

    dongList.forEach(function(dong) {{
        var color  = dongColorMap[dong] || '#888';
        dongStates[dong] = true;

        var item = document.createElement('div');
        item.className = 'legend-item';

        var dot = document.createElement('div');
        dot.className = 'legend-dot';
        dot.style.background = color;

        var label = document.createElement('span');
        label.className = 'legend-label';
        label.textContent = dong;

        // 토글 스위치
        var sw = document.createElement('label');
        sw.className = 'toggle-switch';
        sw.style.setProperty('--dong-color', color);

        var cb = document.createElement('input');
        cb.type    = 'checkbox';
        cb.checked = true;
        cb.addEventListener('change', (function(d) {{
            return function() {{
                dongStates[d] = this.checked;
                dongCircles[d].forEach(function(c) {{
                    c.setMap(this.checked ? map : null);
                }}.bind(this));
                updateAllBtn();
            }};
        }})(dong));

        var slider = document.createElement('span');
        slider.className = 'toggle-slider';

        sw.appendChild(cb);
        sw.appendChild(slider);

        item.appendChild(dot);
        item.appendChild(label);
        item.appendChild(sw);
        legendEl.appendChild(item);
    }});

    // ── 전체 켜기/끄기 버튼 ─────────────────────────────────
    var allOn = true;
    var btn   = document.getElementById('toggle-all-btn');

    function updateAllBtn() {{
        var anyOn = dongList.some(function(d) {{ return dongStates[d]; }});
        allOn = anyOn;
        btn.textContent = anyOn ? '전체 끄기' : '전체 켜기';
    }}

    btn.addEventListener('click', function() {{
        allOn = !allOn;
        // 체크박스 상태 업데이트
        var checkboxes = legendEl.querySelectorAll('input[type=checkbox]');
        checkboxes.forEach(function(cb, i) {{
            cb.checked = allOn;
            var dong = dongList[i];
            dongStates[dong] = allOn;
            dongCircles[dong].forEach(function(c) {{
                c.setMap(allOn ? map : null);
            }});
        }});
        btn.textContent = allOn ? '전체 끄기' : '전체 켜기';
    }});
</script>
</body>
</html>"""

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"🗺️  HTML 생성 완료: {OUTPUT_HTML}")


# ── 서버 자동 시작 ────────────────────────────────────────
_server_started = False

def open_map_in_browser(port=8000):
    global _server_started

    html_dir = os.path.dirname(os.path.abspath(OUTPUT_HTML))
    url      = f"http://localhost:{port}/naver_store_map.html"

    if not _server_started:
        def _serve():
            os.chdir(html_dir)
            handler = http.server.SimpleHTTPRequestHandler
            handler.log_message = lambda *args: None
            
            # 🛠️ [개선] 이미 할당되어 대기 상태인 주소/포트를 즉시 재사용하도록 설정
            socketserver.TCPServer.allow_reuse_address = True
            
            try:
                with socketserver.TCPServer(("", port), handler) as httpd:
                    print(f"🌐 로컬 서버 시작: http://localhost:{port}")
                    print(f"🔗 브라우저 자동 오픈: {url}")
                    print("🌐 서버 실행 중 | 런처: [서버 종료] 버튼 | 터미널: Ctrl+C")
                    
                    # 🛠️ [개선] serve_forever() 대신, Windows에서 키보드 인터럽트를 감지할 수 있도록 루프 구현
                    while True:
                        httpd.handle_request()
                        time.sleep(0.1) # CPU 과점유 방지 및 즉각적인 인터럽트 체크 용도
                        
            except (KeyboardInterrupt, SystemExit):
                print("\n👋 서버를 종료합니다.")
                sys.exit(0)

        thread = threading.Thread(target=_serve, daemon=True)
        thread.start()
        _server_started = True

        # 메인 프로세스의 즉각적 출력을 위해 타이머 비동기 실행 유지
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()