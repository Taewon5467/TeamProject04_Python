# 배달 로봇을 위한 최적의 집결 스팟 선정 프로젝트

## 📌프로젝트 선정 이유 (Project Background)

최근 배달 플랫폼 시장의 급격한 성장과 함께, 가맹점과 소비자 모두가 체감하는 시장의 페인 포인트(Pain Point)를 해결하고자 본 프로젝트를 기획하게 되었습니다.

### 1. 배달 비용 상승 및 지연 문제
* **가맹점 수익성 악화 및 소비자 부담 가중:** 지속적인 배달비 상승은 중소 가맹점의 마진 구조를 악화시킬 뿐만 아니라, 최종 소비자의 경제적 부담을 늘려 플랫폼 이탈을 유도하고 있습니다.
* **서비스 신뢰도 저하:** 고질적인 배달 지연 문제는 소비자 만족도를 떨어뜨려 플랫폼 및 가맹점에 대한 전반적인 신뢰도를 저하시키고, 이는 곧 가맹점의 장기적인 매출 감소로 이어집니다.

### 2. 환경적 요인에 따른 라이더 수급 불균형
* **일시적 공급 절벽 현상:** 주문 수요가 폭증하는 피크타임(점심/저녁)이나 우천·폭설 등 기상 조건이 악화될 때, 라이더 공급이 일시적으로 급감하는 현상이 발생합니다.
* **예측 불가능성 해소의 필요성:** 이러한 실시간 수급 불균형은 배달 지연과 비용 상승의 악순환을 심화시키므로, 데이터 기반의 효율적인 수급 예측 및 매칭 시스템이 필수적입니다.

## 🛠️기술 스택 (Tech Stack)

- **Language:** Python 3.10+
- **Data Analysis & ML:** Pandas, NumPy, Scikit-learn, XGBoost, LightGBM
- **Data Crawling:** Selenium, Requests, BeautifulSoup
- **Visualization:** Matplotlib, Seaborn
- **Environment & Tools:** Git, GitHub, VS Code

## 💡주요 기능 (Key Features)

- **배달 데이터 자동 수집:** Selenium을 활용하여 특정 지역(예: 요기요 등)의 가맹점 정보 및 배달 데이터 크롤링
- **수급 및 트렌드 분석:** 기상 조건, 시간대별 요인을 반영한 데이터 전처리 및 탐색적 데이터 분석(EDA)
- **배달 지연/수요 예측 모델:** 머신러닝 라이브러리(Scikit-learn, XGBoost 등)를 활용한 피크타임 및 기상 악화 시 수요/공급 예측
- **시각화 리포트:** Matplotlib과 Seaborn을 활용하여 시간대별 라이더 수급 불균형 트렌드 시각화

## 📂프로젝트 구조 (Directory Structure)

```text
├── DataSet/                              # 수집된 데이터 및 전처리 데이터 폴더
├── KANG/
│   ├── CLEAN.py                          # 데이터 전처리
│   ├── Data_Algorithm_Evaluation.py      # 머신러닝 모델링 및 성능평가
│   ├── Data_BaseLine.py                  # 베이스라인
│   ├── Data_Evaluation.py                # 데이터 평가
│   ├── Data_Model.py                     # 머신러닝 모델 및 하이퍼파라미터 튜닝
│   └── MERGE.py                          # 데이터셋 통합 및 결측치 처리
├── TEST/
│   ├── clustering evaluation.py          # 군집화 모델링 및 성능평가
│   ├── clustering.py                     # 거점 위치 찾기
│   ├── maping.py                         # 주소 매핑
│   ├── Test.py                           # 네이버 API 인증 테스트
│   └── WepCrolling.py                    # 요기요 웹 크롤링
├── .gitignore                            # 깃허브 업로드 제외 설정 파일
├── clustering.py                         # DCF 군집화 및 거점 산출
├── config.py                             # 모든 상수 관리
├── geocoding.py                          # 네이버 API 좌표 변환
├── launcher.py                           # 실행 파일
├── main.py                               # 메인 파일
├── prediction.py                         # 시뮬레이션
├── requirements.txt                      # 의존성 라이브러리 목록
├── visualization.py                      # 네이버 지도 시각화
└── README.md                             # 프로젝트 안내 문서
```
### ⚙️설치 및 실행 방법 (Installation & Usage)

#### 1) 네이버 클라우드 API 설정
1. [네이버 클라우드 플랫폼](https://ncloud.com/)에 접속하여 회원가입 및 로그인을 진행합니다.
2. **Services** 메뉴에서 **Application Service** -> **Maps**를 선택하여 이용 신청을 합니다.
3. 좌측 **AI·NAVER API** 탭의 **Application** 메뉴를 클릭한 뒤 **Application 등록**을 진행합니다.
4. API 선택 항목에서 **Dynamic Map**과 **Geocoding**을 체크합니다.
5. **Web 서비스 URL** 입력창에 `http://localhost` 및 `http://localhost:8000`을 추가합니다.
6. 하단의 등록 버튼을 눌러 API 신청을 완료합니다.

#### 2) 프로젝트 환경 구축 및 실행
1. 아래 명령어를 통해 저장소를 클론(다운로드)하거나, 깃허브에서 ZIP 파일로 다운로드합니다.
   ```bash
   git clone [https://github.com/Taewon5467/TeamProject04_Python.git](https://github.com/Taewon5467/TeamProject04_Python.git)
2. TeamProject04_Python 폴더 위치로 CMD 를 열어준 뒤, `pip install -r requirements.txt`를 입력하여 프로젝트에 필요한 필수 라이브러리들을 설치합니다.
3. TeamProject04_Python 루트 폴더 아래에 .env 파일을 새로 생성합니다.
4. 발급받은 네이버 API 키를 .env 파일에 아래와 같은 형식으로 입력하고 저장합니다.
```
NAVER_CLIENT_ID = '발급받은 Client ID'
NAVER_CLIENT_SECRET = '발급받은 Client Secret'
```
5. 모든 설정이 완료되었다면, 터미널(CMD)에 아래 명령어를 입력하여 프로그램을 실행합니다.
   ```bash
   python launcher.py
   ```

## 📊 분석 및 예측 결과 (Results)
MAE  9.0818  

RMSE 14.3436  

R² Score  0.8930
