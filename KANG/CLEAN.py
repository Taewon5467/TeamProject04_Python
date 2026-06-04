import pandas as pd
import os

clean_files = [
    "ansan_delivery_2019_clean.csv",
    "ansan_delivery_2020_clean.csv",
    "ansan_delivery_2021_clean.csv"
]

dfs = []
for f in clean_files:
    if os.path.exists(f):
        dfs.append(pd.read_csv(f))
    else:
        alt_path = os.path.join("DataSet", f)
        if os.path.exists(alt_path):
            dfs.append(pd.read_csv(alt_path))
        else:
            parent_alt_path = os.path.join("..", "DataSet", f)
            if os.path.exists(parent_alt_path):
                dfs.append(pd.read_csv(parent_alt_path))
            else:
                print(f"⚠️ {f} 파일을 찾을 수 없습니다.")

if dfs:
    df_all = pd.concat(dfs, ignore_index=True)
    df_all['월'] = df_all['날짜'].apply(lambda x: int(x.split('-')[0]))
    df_all['일'] = df_all['날짜'].apply(lambda x: int(x.split('-')[1]))
    df_all['월'] = df_all['월'].astype('category')
    df_all['시간'] = df_all['시간'].astype('category')
    df_all['요일'] = df_all['요일'].astype('category')
else:
    df_all = pd.DataFrame(columns=['날짜', '시간', '요일', '기온', '강수량', '적설', '시정', '시도', '시군구', '업종', '주문건수', '월', '일'])