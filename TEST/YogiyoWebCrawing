from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd
import os
import re

location = input("지역 입력: ")

chrome_options = Options()
chrome_options.add_experimental_option("detach", True) 
chrome_options.add_argument("--disable-blink-features=AutomationControlled")

chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
prefs = {"profile.managed_default_content_settings.images": 2}
chrome_options.add_experimental_option("prefs", prefs)

#url 입력
driver = webdriver.Chrome(options=chrome_options)
driver.implicitly_wait(10)
url = "https://www.yogiyo.co.kr"

categories = [
    "1인분 주문", "프랜차이즈", "치킨", "피자/양식", "중국집", 
    "한식", "일식/돈까스", "족발/보쌈", "야식", "분식", 
    "카페/디저트", "편의점/마트"
]

file_name = f"{location}_통합_가게정보.csv"

if os.path.exists(file_name):
    os.remove(file_name)

#중복을 걸러내기 위한 '수집 완료된 가게 이름' 저장소 (집합 구조로 검색 속도가 빠름)
seen_restaurants = set()
try:
    driver.get(url)
    driver.maximize_window()
    # 검색 단계
    wait = WebDriverWait(driver, 10)
    search = wait.until(EC.presence_of_element_located((By.NAME, "address_input")))
    
    # 1. 기존 주소 지우기
    search.send_keys(Keys.CONTROL + "a")
    search.send_keys(Keys.BACKSPACE)
    time.sleep(0.5) 
    
    # 2. 새로운 지역 입력
    search.send_keys(location)
    time.sleep(1)
    
    # 3. '검색' 버튼 클릭
    search_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[text()='검색']")))
    search_btn.click()
    time.sleep(1.5) # 검색 결과 드롭다운 뜰 때까지 대기
    
    # 4. 첫 번째 실제 장소 클릭 ("현재 위치로 설정합니다" 건너뛰기)
    dropdown_items = wait.until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="search"]/div/form/ul/li/a')))
    
    for item in dropdown_items:
        # 텍스트에 '현재 위치'가 포함되지 않은 첫 번째 항목을 클릭합니다.
        if "현재 위치" not in item.text:
            print(f"📍 설정된 기준 주소: {item.text}")
            item.click()
            break
    
# 2. 카테고리별 순회
    for category in categories:
        print(f"\n=======================================")
        print(f" [{category}] 카테고리 수집을 시작합니다.")
        print(f"=======================================\n")
        
        try:
            category_tab = wait.until(EC.element_to_be_clickable((By.XPATH, f"//span[contains(text(), '{category}')]")))
            driver.execute_script("arguments[0].click();", category_tab)
            time.sleep(2) 
            
        except Exception as e:
            print(f"'{category}' 카테고리를 찾을 수 없어 다음으로 넘어갑니다.")
            continue

        print(f"[{category}] 전체 가게 목록을 로딩합니다...")
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break 
            last_height = new_height

        items = driver.find_elements(By.CSS_SELECTOR, "div.item.clearfix")
        count = len(items)
        print(f"🤖 [{category}] 총 {count}개 탐색을 시작합니다.")

       # --- [수정 및 추가된 부분 🚨 3] 클릭을 배제하고 Angular DOM에서 고유 URL 100% 정확하게 수집 ---
        restaurants_to_visit = []
        
        for item in items:
            try:
                name_element = item.find_element(By.CSS_SELECTOR, ".restaurant-name")
                res_name = name_element.text.strip()
                
                if not res_name or res_name in seen_restaurants:
                    continue
                    
                # 💡 핵심 수정: HTML 글자를 뒤지는 대신, 요기요의 Angular 데이터 객체에서 ID를 직접 꺼내옵니다.
                res_id = driver.execute_script("""
                    var elem = arguments[0];
                    var scope = angular.element(elem).scope();
                    if(scope && scope.restaurant) {
                        return scope.restaurant.id;
                    }
                    return null;
                """, item)

                if res_id:
                    # 완벽하게 추출된 ID로 URL을 조립합니다.
                    res_url = f"https://www.yogiyo.co.kr/mobile/#/{res_id}/"
                    restaurants_to_visit.append({"name": res_name, "url": res_url})
                else:
                    print(f"⚠️ '{res_name}'의 고유 ID를 내부 데이터에서 찾을 수 없습니다.")
                    
            except Exception as e:
                pass

        print(f"✅ URL 수집 완료! 총 {len(restaurants_to_visit)}개의 가게 세부 정보로 직접 이동하여 주소를 캡처합니다.")
        
        results = []
        
        for idx, target in enumerate(restaurants_to_visit):
            res_name = target["name"]
            res_url = target["url"]
            
            print(f"[{idx+1}/{len(restaurants_to_visit)}] {res_name} 수집 중...")
            
            try:
                # 직접 URL로 이동
                time.sleep(1.5)
                driver.get(res_url)
                
                # 가끔 뜨는 "배달 불가 지역" 알림창(Alert) 무시 처리
                try:
                    alert = driver.switch_to.alert
                    alert.dismiss()
                except:
                    pass

                # '정보' 탭 클릭
                info_tab = wait.until(EC.element_to_be_clickable((By.XPATH, "//*[text()='정보']")))
                time.sleep(1)
                driver.execute_script("arguments[0].click();", info_tab)
                time.sleep(0.5)
                # 주소 가져오기
                address_element = wait.until(EC.visibility_of_element_located(
                    (By.XPATH, "//*[text()='주소']/following-sibling::*[1]")
                ))
                
                address = address_element.get_attribute("innerText").strip()
                if not address:
                    address = address_element.get_attribute("textContent").strip()

                if location in address:
                    results.append({"카테고리": category, "가게명": res_name, "주소": address})
                    seen_restaurants.add(res_name) 
                    print(f"  ✅ '{location}' 확인됨! (주소: {address})")
                else:
                    print(f"  ❌ '{location}' 미포함. 제외합니다. (주소: {address})")
                
            except Exception as e:
                # [수정됨] 왜 오류가 났는지 확인할 수 있도록 URL과 에러 메시지를 출력합니다.
                print(f"  ⚠️ 오류 발생! 접속 시도한 주소: {res_url}")
                # print(f"  ⚠️ 상세 에러: {e}") # 필요 시 주석을 해제하면 아주 긴 에러 로그를 볼 수 있습니다.

        # 카테고리 완료 후 저장
        if results:
            df = pd.DataFrame(results)
            if not os.path.exists(file_name):
                df.to_csv(file_name, index=False, encoding='utf-8-sig', mode='w')
            else:
                df.to_csv(file_name, index=False, encoding='utf-8-sig', mode='a', header=False)
                
            print(f"\n💾 [{category}] 수집 완료! 통합 파일 '{file_name}'에 누적 저장되었습니다.\n")
        else:
            print(f"\nℹ️ [{category}] 새로 수집할 조건에 맞는 가게가 없습니다.\n")

finally:
    print(f"\n🎉 모든 작업이 종료되었습니다! 총 {len(seen_restaurants)}개의 중복 없는 가게 데이터가 '{file_name}'에 저장되었습니다.")
    driver.quit()