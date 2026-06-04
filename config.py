# 모든 상수 중앙 관리
import os
from dotenv import load_dotenv

load_dotenv()

NAVER_CLIENT_ID     = os.environ.get("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET")

if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
    raise EnvironmentError("❌ .env 파일에 API 키를 입력해 주세요.")

# 파일 경로
FILE_RAW        = r'DataSet\상록구 본오동_통합_가게정보.csv'
FILE_GEO_CACHED = r'DataSet\상록구 본오동_통합_가게정보_좌표추가.csv'
PIPELINE_PATH   = 'lgbm_delivery_pipeline.pkl'
OUTPUT_HTML     = 'naver_store_map.html'
OUTPUT_SIM_DIR  = 'simulation_results'

# DCF 파라미터
DELIVERY_TARGET_MINUTES = 10
SPEED_FACTOR            = 15.0
COMPLEXITY_FACTOR       = 1.35
DETOUR_FACTOR           = 2.22

# API
MAX_THREADS = 15

# 시뮬레이션 기본값
DEFAULT_DATE    = "2026-06-15"
DEFAULT_WEEKDAY = "월요일"
DEFAULT_TEMP    = 40
DEFAULT_RAIN    = 9.0
DEFAULT_SNOW    = 0.0
DEFAULT_VIS     = 20000.0