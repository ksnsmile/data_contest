# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 14:51:32 2024

@author: sn714
"""

import pandas as pd

# 파일 경로
file_path ='C:/Users/sn714/OneDrive/바탕 화면/data_contest/23.12월 주요 국가산업단지 산업동향(공시용) (1).xlsx'

# 엑셀 파일의 모든 시트 이름을 가져오기
xls = pd.ExcelFile(file_path)


sheet=['표4 단지별 생산','표8 단지별 고용','표6 단지별 수출']
# 빈 데이터프레임 생성




단지별_생산=pd.read_excel(file_path,sheet_name=sheet[0])
단지별_고용=pd.read_excel(file_path,sheet_name=sheet[1])
단지별_수출=pd.read_excel(file_path,sheet_name=sheet[2])

selected_columns = 단지별_생산.iloc[:, [0, -1]]
selected_columns.columns=selected_columns.iloc[1]
selected_columns= selected_columns.drop([0, 1])
selected_columns= selected_columns.drop([40,41])
생산파일=selected_columns.copy()
생산파일 = 생산파일.reset_index()
생산파일.drop(columns=['산업단지','index'],inplace=True)
생산파일.rename(columns={'증감률\n(전월대비)':'생산_증감률'},inplace=True)
selected_columns_1=단지별_고용.iloc[:, [0, -1]]
selected_columns_1.columns=selected_columns_1.iloc[1]
selected_columns_1= selected_columns_1.drop([0, 1])
selected_columns_1= selected_columns_1.drop([2])
selected_columns_1= selected_columns_1.drop([41])
고용파일=selected_columns_1.copy()
고용파일 = 고용파일.reset_index()

고용파일.drop(columns=['산업단지','index'],inplace=True)
고용파일.rename(columns={'전월대비':'고용_증감률'},inplace=True)
selected_columns_2 = 단지별_생산.iloc[:, [0, -1]]
selected_columns_2.columns=selected_columns_2.iloc[1]
selected_columns_2= selected_columns_2.drop([0, 1])
selected_columns_2= selected_columns_2.drop([40,41])
수출파일=selected_columns_2.copy()
수출파일 =수출파일.reset_index()
수출파일.drop(columns=['index'],inplace=True)
수출파일.rename(columns={'증감률\n(전월대비)':'수출_증감률'},inplace=True)







combined_df = pd.concat([수출파일, 고용파일,생산파일], axis=1)


combined_df.isna().sum()





print((combined_df == 'X').sum())



combined_df = combined_df.loc[~((combined_df['수출_증감률'] == 'X') | 
                                (combined_df['고용_증감률'] == 'X') | 
                                (combined_df['생산_증감률'] == 'X'))]


df=pd.read_excel('C:/Users/sn714/OneDrive/바탕 화면/data_contest/공장창고등(매매)_실거래가 공개 내역(국토교통부 제공) (1).xlsx')

df.columns=df.iloc[11]
df = df.drop(index=range(0, 12))



df=df.reset_index()
df.columns
df.drop(columns=['index', 'NO'],inplace=True)

df_change=df.copy()
df_change.columns


df_change.drop(columns=['유형', '지번','용도지역','층', '매수자', '매도자', '계약년월', '계약일', '지분구분', '건축년도',
'해제사유발생일', '거래유형', '중개사소재지'],inplace=True)



df_change.to_excel("파일1.xlsx",index=False)
combined_df.to_excel("파일2.xlsx",index=False)

df_change['시군구'].unique()


List=combined_df['산업단지'].unique()

List=list(List)


# =============================================================================
# filter_data = df_change[df_change['시군구'].str.contains('|'.join(List))]
# 
# 
# 
# 
# 
# 
# df_change['키'] = df_change['시군구'].apply(lambda x: next((keyword for keyword in combined_df['산업단지'] if keyword in x), None))
# combined_df['키'] = combined_df['산업단지']
# 
# # merge
# merged_df = pd.merge(df_change, combined_df, on='키', how='left').drop('키', axis=1)
# =============================================================================


df_change['키'] = df_change['시군구'].apply(lambda x: next((keyword for keyword in List if keyword in x), None))

# combined_df에서도 키 열 생성
combined_df['키'] = combined_df['산업단지']

# merge
merged_df = pd.merge(df_change, combined_df, on='키', how='left').drop('키', axis=1)


merged_df.isna().sum()

import pandas as pd

전기 = pd.read_excel('C:/Users/sn714/OneDrive/바탕 화면/data_contest/산업분류별 월별 전력사용량_20240607 (1).xls')

전기.columns=전기.iloc[26]
전기.drop(index=list(range(0,27)),inplace=True)
전기=전기.reset_index()
전기.columns
전기.drop(columns=['index','년월','시군구','고객호수(호)'],inplace=True)






merged_df['키'] = merged_df['시군구'].apply(lambda x: next((keyword for keyword in 전기['시구'] if keyword in x), None))
전기['키'] = 전기['시구']

# merge
merged_df = pd.merge(merged_df, 전기, on='키', how='left').drop('키', axis=1)



df_1=pd.read_excel('C:/Users/sn714/OneDrive/바탕 화면/data_contest/지역 위치별 건축면적.xlsx')
df_1.columns
df_change_1=df_1.drop(columns=['공장관리번호', '회사명','대표자', '행정기관'])


df_2=pd.read_excel('C:/Users/sn714/OneDrive/바탕 화면/data_contest/지역별 업종.xlsx')
df_2.columns
df_change_2=df_2.drop(columns=['공장관리번호', '회사명','대표자', '행정기관'])






df_merged=pd.merge(df_change_2,df_change_1,on='도로명주소',how='left')




df_merged.isna().sum()

df_merged.dropna(inplace=True)


data_1=df_merged.copy()
data_2=merged_df.copy()




# 키 생성
def find_matching_keywords(row, keywords):
    matching_keywords = [keyword for keyword in keywords if keyword in row]
    return matching_keywords[0] if matching_keywords else None

data_2['키'] = data_2['시군구'].apply(lambda x: find_matching_keywords(x, data_1['도로명주소']))
data_1['키'] = data_1['도로명주소']

# merge
merged_df_2 = pd.merge(data_2, data_1, on='키', how='left').drop('키', axis=1)

def find_matching_keyword(row, keywords):
    for keyword in keywords:
        if keyword in row:
            return keyword
    return None

# data_2에 키 생성
data_2['키'] = data_2['시군구'].apply(lambda x: find_matching_keyword(x, data_1['도로명주소']))

# data_1에 키 생성 (자체 키)
data_1['키'] = data_1['도로명주소']

# merge
merged_df_2 = pd.merge(data_2, data_1, on='키', how='left').drop('키', axis=1)

matches = []
for index, row in data_2.iterrows():
    match = data_1[data_1['도로명주소'].str.contains(row['시군구'])]
    if not match.empty:
        match = match.copy()
        match['키'] = row['시군구']
        matches.append(match)
    else:
        matches.append(pd.DataFrame({'도로명주소': [None], '인구': [None], '키': [row['시군구']]}))

# 모든 매치를 하나의 데이터프레임으로 병합
matches_df = pd.concat(matches, ignore_index=True)

# '키' 열을 사용하여 조인
merged_df = pd.merge(data_2, matches_df, left_on='시군구', right_on='키', how='left').drop('키', axis=1)



data_2['키'] = data_2['시군구'].apply(lambda x: next((keyword for keyword in data_1['도로명주소'] if keyword in x), None))
data_1['키'] = data_1['도로명주소']

# merge
merged_df__v2 = pd.merge(merged_df, 전기, on='키', how='left').drop('키', axis=1)



data_1.to_excel("데이터통합_1.xlsx",index=False)
data_2.to_excel("데이터통합_2.xlsx",index=False)



