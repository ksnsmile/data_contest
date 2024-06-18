# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 09:10:46 2024

@author: sn714
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd

# API 요청 정보
service_key = 'xq2BPOH3pyG34ZazOnuazbXIp%2FdG4FvmO7uyIk6CxWiW%2Fwj0yewdq5ICepR0ZEMsLsFl2jaHNse%2FtkGPiXoPEg%3D%3D'
num_of_rows = '100'
fctry_manage_no = ''
rprsntv_nm = ''
main_product_cn = ''
adres_code = ''
adres = ''
req_type = 'XML'

# 페이지와 회사명 리스트
pages = list(range(1, 11))
companies = ['기아', '현대', '삼성', 'LG', '롯데', '쌍용', 'SK']

# 결과를 저장할 리스트
data = []

# 모든 페이지와 회사명에 대해 API 요청
for company in companies:
    for page in pages:
        url = (
            f'http://apis.data.go.kr/B550624/fctryRegistLndpclInfo/getFctryLndpclService'
            f'?serviceKey={service_key}'
            f'&pageNo={page}'
            f'&numOfRows={num_of_rows}'
            f'&fctryManageNo={fctry_manage_no}'
            f'&cmpnyNm={company}'
            f'&rprsntvNm={rprsntv_nm}'
            f'&mainProductCn={main_product_cn}'
            f'&adresCode={adres_code}'
            f'&adres={adres}'
            f'&type={req_type}'
        )

        response = requests.get(url)
        print(f"Fetching data for company: {company}, page: {page}, status code: {response.status_code}")
        
        if response.status_code == 200:
            # XML 파일 파싱
            root = ET.fromstring(response.content)

            # 모든 태그를 추출하여 컬럼명으로 사용
            for item in root.findall('.//item'):
                row = {}
                for elem in item:
                    row[elem.tag] = elem.text
                data.append(row)

# DataFrame 생성
df = pd.DataFrame(data)

# DataFrame 출력
print(df)

df.columns

df.rename(columns={'fctryManageNo':'공장관리번호',
                   'cmpnyNm':'회사명','rnAdres':'도로명주소',
                   'rprsntvNm':'대표자','cvplChrgOrgnztNm':'행정기관',
                   'cmpnyTelno':'회사전화번호','fctryLndpclAr':'용지면적',
                   'fctryDongBuldAr':'건축면적', 'spfcSeCodeNm':'용도지역'},inplace=True)
                   


df.drop(columns=['irsttNm'],inplace=True)


df.isna().sum()

df.drop(columns=['회사전화번호'],inplace=True)

df.drop(columns=['용지면적'],inplace=True)


df.dropna(inplace=True)



df.to_excel('지역 위치별 건축면적.xlsx',index=False)













