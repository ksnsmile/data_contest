# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:16:59 2024

@author: sn714
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 데이터 로드 및 전처리
data = pd.read_csv("최종데이터.csv")

# 컬럼명 변경 및 불필요한 컬럼 제거
data.rename(columns={'평균판매단가(원/kWh)': '전력단가(원/kwh)'}, inplace=True)
data.drop(columns=['용도지역', '업종명'], inplace=True)

# 전처리된 데이터 저장
data.to_csv("최종데이터_1.csv", index=False)

# 클러스터링 전 확인 작업 - 원본 데이터의 박스플롯 시각화
fig, ax = plt.subplots()
ax.boxplot([data['고용인원'], data['건축면적'], data['전력단가(원/kwh)']], sym='rs')  # rs = red square
plt.xticks([1, 2, 3], ['고용인원', '건축면적', '전력단가(원/kwh)'])
plt.title('Original Data Boxplot')
plt.show()

# Standard 스케일링 적용
standard_scaler = StandardScaler()
data_standard_scaled = standard_scaler.fit_transform(data[['고용인원', '건축면적', '전력단가(원/kwh)']])

# 스케일링된 데이터의 박스플롯 시각화
fig, ax = plt.subplots()
ax.boxplot(data_standard_scaled, sym='rs')  # rs = red square
plt.xticks([1, 2, 3], ['고용인원_scaled', '건축면적_scaled', '전력단가(원/kwh)_scaled'])
plt.title('Standard Scaled Data Boxplot')
plt.show()


X_features_scaled=data_standard_scaled


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

## 4-2. Determine the number of clusters(k) - elbow method (엘보 방법) 

distortions = []   # store distortion values (inertia_) in a List

for i in range(1, 11):    # k : 1 ~ 10
    kmeans_i = KMeans(n_clusters=i, random_state=0)  
    kmeans_i.fit(X_features_scaled)   
    distortions.append(kmeans_i.inertia_)
    
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()



### => Conclusion : k = 3

## 4(a)-3. Generate the Model - k=3

### kmeans_3 : instance of KMeans Class
kmeans_3 = KMeans(n_clusters=3, random_state=0)

## 4(a)-4. Train the model - compute k-means clustering

### fit(x)
kmeans_3.fit(X_features_scaled)

## 4(a)-5. Predict the closest cluster each sample in X belongs to

### predict(x)
Y_labels = kmeans_3.predict(X_features_scaled)

### Y_labels = kmeans.fit_predict(X_features_scaled) 

## 4(a)-6. Add labels to customer_df

data_3 = data.copy()
data_3['ClusterLabel'] = Y_labels
data_3.head()


# 5. Evaluate & Visualize the analysis results

## 5(a)-1. Evaluate  - silhouette analysis (실루엣 분석)

score_samples = silhouette_samples(X_features_scaled, Y_labels, metric='euclidean')
score_samples.shape
score_samples
data_3['SilhoutteCoeff'] = score_samples

average_score = silhouette_score(X_features_scaled, Y_labels)
average_score
data_3.groupby('ClusterLabel')['SilhoutteCoeff'].mean()





kmeans_4 = KMeans(n_clusters=4, random_state=0)

## 4(b)-4. Train the model - compute k-means clustering

### fit(x)
kmeans_4.fit(X_features_scaled)

## 4(b)-5. Predict the closest cluster each sample in X belongs to

### predict(x)
Y_labels = kmeans_4.predict(X_features_scaled)

### Y_labels = kmeans.fit_predict(X_features_scaled) 

## 4(b)-6. Add labels to customer_df

data_4 = data.copy()
data_4['ClusterLabel'] = Y_labels
data_4.head()


# 5. Evaluate & Visualize the analysis results

## 5(b)-1. Evaluate  - silhouette analysis (실루엣 분석)

score_samples = silhouette_samples(X_features_scaled, Y_labels, metric='euclidean')
score_samples.shape
score_samples
data_4['SilhoutteCoeff'] = score_samples

average_score = silhouette_score(X_features_scaled, Y_labels)
average_score
data_4.groupby('ClusterLabel')['SilhoutteCoeff'].mean()





























