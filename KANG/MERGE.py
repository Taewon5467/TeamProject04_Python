import pandas as pd
import os

file_paths = {
    2019: r"DataSet\KGU_3rd_ORIGIN_KGUDSTNORDER_20190101000000.csv",
    2020: r"DataSet\KGU_3rd_ORIGIN_KGUDSTNORDER_20200101000000.csv",
    2021: r"DataSet\KGU_3rd_ORIGIN_KGUDSTNORDER_20210101000000.csv"
}

weather_paths = {
    2019: r"DataSet\OBS_ASOS_TIM_20190812.csv",
    2020: r"DataSet\OBS_ASOS_TIM_20200112.csv",
    2021: r"DataSet\OBS_ASOS_TIM_20210107.csv"
}

day_map = {0: '월요일', 1: '화요일', 2: '수요일', 3: '목요일', 4: '금요일', 5: '토요일', 6: '일요일'}

for year, path in file_paths.items():
    if not os.path.exists(path):
        print(f"⚠️ 주문 데이터 '{path}' 파일이 현재 폴더에 없습니다. 건너끕니다.")
        continue
        
    w_path = weather_paths[year]
    if not os.path.exists(w_path):
        print(f"⚠️ 기상 데이터 '{w_path}' 파일이 현재 폴더에 없습니다. 건너끕니다.")
        continue

    print(f"📦 [{year}년] 주문 + 기상 데이터 통합 전처리 시작...")
    chunks = []
    
    for chunk in pd.read_csv(path, header=None, chunksize=200000):
        chunk.columns = ['날짜', '시간', '업종', '시도', '시군구', '주문건수']
        filtered_chunk = chunk[(chunk['시도'] == '경기도') & (chunk['시군구'] == '안산시 상록구')].copy()
        chunks.append(filtered_chunk)
        
    df_year = pd.concat(chunks, ignore_index=True)
    df_year['업종'] = '음식'
    df_year['날짜'] = pd.to_datetime(df_year['날짜'])
    
    if year == 2019:
        df_year = df_year[~df_year['날짜'].dt.month.isin([5, 6, 7])].reset_index(drop=True)
        print("✂️ [편향 제거] 2019년 5~7월 데이터를 제외했습니다.")

    df_year['요일'] = df_year['날짜'].dt.weekday.map(day_map)

    try:
        df_weather = pd.read_csv(w_path, encoding='cp949')
    except UnicodeDecodeError:
        df_weather = pd.read_csv(w_path, encoding='utf-8')
        
    df_weather['일시'] = pd.to_datetime(df_weather['일시'])
    df_weather['날짜'] = df_weather['일시'].dt.normalize()
    df_weather['시간'] = df_weather['일시'].dt.hour
    
    df_weather = df_weather[['날짜', '시간', '기온(°C)', '강수량(mm)', '적설(cm)', '시정(10m)']].copy()
    df_weather.columns = ['날짜', '시간', '기온', '강수량', '적설', '시정']
    
    df_weather['강수량'] = df_weather['강수량'].fillna(0.0)
    df_weather['적설'] = df_weather['적설'].fillna(0.0)
    df_weather['기온'] = df_weather['기온'].ffill().bfill()
    df_weather['시정'] = df_weather['시정'].ffill().bfill()
    
    df_merged = pd.merge(df_year, df_weather, on=['날짜', '시간'], how='inner')
    df_merged['날짜'] = df_merged['날짜'].dt.strftime('%m-%d')
    
    df_aggregated = df_merged.groupby(
        ['날짜', '시간', '요일', '기온', '강수량', '적설', '시정', '시도', '시군구', '업종'], 
        as_index=False
    )['주문건수'].sum()
    
    df_aggregated = df_aggregated.sort_values(by=['날짜', '시간']).reset_index(drop=True)
    
    output_filename = f"ansan_delivery_{year}_clean.csv"
    df_aggregated.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"✨ [{year}년] 주문+날씨 통합 및 연도 제거 완료 -> 📂 {output_filename} (최종 행 수: {df_aggregated.shape[0]:,}개)\n")

print("🎉 연도 데이터가 완벽하게 배제된 연도별 파일 작성이 완료되었습니다!")