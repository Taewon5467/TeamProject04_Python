import requests

print("🔍 네이버 API 권한(인증) 테스트 스크립트입니다.")
print("-" * 50)

# 🚨 네이버 콘솔 [인증 정보] 창에서 복사한 값을 아래에 정확히 붙여넣어 보세요.
# 기존에 쓰시던 hymjc5hhjr 말고, 새로 발급받으신 키가 있다면 그걸로 넣어보시는 걸 추천합니다.
CLIENT_ID = "b0h6eca72o" 
CLIENT_SECRET = "PgNEfT7XUtWLKmnEoCpySjVaNk2yS7Jpx8pBrbMv"

# 테스트용 주소 (임의의 정상 주소)
test_address = "경기도 성남시 분당구 정자일로 95" 

url = "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
headers = {
    "X-NCP-APIGW-API-KEY-ID": CLIENT_ID,
    "X-NCP-APIGW-API-KEY": CLIENT_SECRET
}
params = {
    "query": test_address
}

print(f"테스트 주소: '{test_address}'")
print(f"사용된 ID: {CLIENT_ID}")
print("API에 연결 중입니다...\n")

try:
    response = requests.get(url, headers=headers, params=params)
    
    print(f"▶ HTTP 상태 코드: {response.status_code}")
    print(f"▶ 네이버 API 원본 응답: {response.text}")
    print("-" * 50)
    
    if response.status_code == 200:
        print("✅ 성공! API 키와 권한 설정이 완벽하게 정상입니다.")
        print("이제 이 ID와 Secret을 원래 작업하시던 코드에 그대로 덮어쓰시면 됩니다.")
        
    elif response.status_code == 401 and "210" in response.text:
        print("❌ 실패! (Error 210: Permission Denied)")
        print("이 키는 Geocoding 권한이 없는 키입니다.")
        print("[해결책] 네이버 콘솔에서 [+ Application 등록]을 눌러 아예 새로운 앱을 하나 파시고,")
        print("Geocoding을 체크한 뒤 나오는 '새로운 ID/Secret'을 여기에 넣고 다시 테스트해 보세요.")
        
    else:
        print("⚠️ 기타 오류 발생. 위에 출력된 원본 응답 내용을 확인해 주세요.")

except Exception as e:
    print(f"네트워크 요청 중 에러가 발생했습니다: {e}")