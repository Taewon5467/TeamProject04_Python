import pandas as pd

file_path = '한국부동산원_공동주택 단지 식별정보_기본정보_20250918.csv'
try:
    df = pd.read_csv(file_path, encoding='cp949')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='utf-8')
    
num_df = df[df['주소'].fillna('').str.contains('상록구')]

print(f"추출된 안산 상록구 공동주책 수: {len(num_df)}개")
print(num_df[['주소', '단지명_공시가격', '동수', '세대수']].head())

output_file = '안산시_상록구_공동주택_기본정보.csv'
num_df.to_csv(output_file, index=False, encoding='cp949')

print(f"\n'{output_file}' 파일이 생성되었습니다!")