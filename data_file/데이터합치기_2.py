# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 14:29:14 2024

@author: sn714
"""

import pandas as pd

df_1=pd.read_excel('C:/Users/sn714/OneDrive/바탕 화면/data_contest/데이터통합_v1.xlsx')
df_2=pd.read_excel('C:/Users/sn714/OneDrive/바탕 화면/data_contest/데이터통합_v2.xlsx')


df_2.columns
df_2_change=df_2.copy()
df_2_change.drop(columns=['시군구','건축물주용도','산업단지','수출_증감률', '고용_증감률', '생산_증감률'],inplace=True)
df_2_change.to_excel("data.xlsx",index=False)

df_1_change=df_1.copy()



df_2_change.isna().sum()

df_2_change.dropna(inplace=True)
df_2_change = df_2_change[df_2_change['도로조건'] != '-']
df_2_change['도로조건'].unique()
df_2_change['도로조건'] = df_2_change['도로조건'].str.replace(r'm미만', '').astype(int)

def clean_road_condition(value):
    if 'm미만' in value:
        return int(value.replace('m미만', '').strip())
    elif 'm이상' in value:
        return 26
    else:
        return None

df_2_change['도로조건'] = df_2_change['도로조건'].apply(clean_road_condition)
df_2_change.columns
df_2_change.drop(columns=['계약구분'],inplace=True)

def remove_hyphen_rows(df):
    # '-' 값을 pd.NA로 대체
    df = df.replace('-', pd.NA)
    # 결측값이 있는 행 제거
    return df.dropna()

# 모든 열에서 '-' 값을 가진 행 제거
df_2_change = remove_hyphen_rows(df_2_change)

numeric_columns = ['도로조건', '전용/연면적(㎡)', '대지면적(㎡)', '거래금액(만원)', '사용량(kWh)', '전기요금(원)', '평균판매단가(원/kWh)']

for col in numeric_columns:
    df_2_change[col] = df_2_change[col].astype(float)


# , 제거 후 숫자형식으로 변환
columns_to_convert = ['거래금액(만원)', '사용량(kWh)', '전기요금(원)', '평균판매단가(원/kWh)']

for column in columns_to_convert:
    df_2_change[column] = df_2_change[column].str.replace(',', '').astype(float)


data_2=df_2_change.groupby("시구").agg({"도로조건":'mean','전용/연면적(㎡)':'mean','대지면적(㎡)':'mean','거래금액(만원)':'mean'
                               ,'사용량(kWh)':'mean','전기요금(원)':'mean','평균판매단가(원/kWh)':'mean'})


data_2.columns

data_2.drop(columns=['도로조건', '전용/연면적(㎡)', '대지면적(㎡)', '거래금액(만원)', '사용량(kWh)', '전기요금(원)'],inplace=True)

# df_1에서 앞에있는 시구만 추출 하여서 합병 작업실시
df_1['시구']=df_1['도로명주소'].apply(lambda x: x.split()[0])


merged_df=pd.merge(df_1,data_2,on='시구',how='left')
merged_df.drop(columns=['시구'],inplace=True)

merged_df.isna().sum()
merged_df.dropna(inplace=True)

merged_df.to_csv("최종데이터.csv",index=False)

