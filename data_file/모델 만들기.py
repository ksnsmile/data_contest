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
data = pd.read_csv("최종데이터_1.csv")

data.columns
data.rename(columns={'도로명주소': 'address','고용인원':'Employees','주생산품':'product','건축면적':'Area','전력단가(원/kwh)':'Power unit price'}, inplace=True)

# 클러스터링 전 확인 작업 - 원본 데이터의 박스플롯 시각화
fig, ax = plt.subplots()
ax.boxplot([data['Employees'], data['Area'], data['Power unit price']], sym='rs')  # rs = red square
plt.xticks([1, 2, 3], ['Employees', 'Area', 'Power unit price'])
plt.title('Original Data Boxplot')
plt.show()

# Standard 스케일링 적용
standard_scaler = StandardScaler()
data_standard_scaled = standard_scaler.fit_transform(data[['Employees', 'Area', 'Power unit price']])

# 스케일링된 데이터의 박스플롯 시각화
fig, ax = plt.subplots()
ax.boxplot(data_standard_scaled, sym='rs')  # rs = red square
plt.xticks([1, 2, 3], ['Employees_scaled', 'Area_scaled', 'Power unit price_scaled'])
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



model_data=data_4.copy()


model_data.drop(columns=['SilhoutteCoeff'],inplace=True)




from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

model_data.columns

Y = model_data['ClusterLabel']
X = model_data[['Employees', 'Area', 'Power unit price']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=156, stratify=Y)

rf = RandomForestClassifier(random_state=156)
rf.fit(X_train, Y_train)
Y_predict = rf.predict(X_test)

# 특정 X 값을 입력했을 때 예측된 클러스터 라벨에 해당하는 도로명주소 출력 함수
def get_addresses_by_prediction(X_new):
    # X_new를 데이터프레임으로 변환하여 피처 이름을 명시
    X_new_df = pd.DataFrame([X_new], columns=['Employees', 'Area', 'Power unit price'])
    X_new_scaled = standard_scaler.transform(X_new_df)
    predicted_label = rf.predict(X_new_scaled)[0]
    
    # 예측된 클러스터 라벨이 유효한지 확인
    if predicted_label not in model_data['ClusterLabel'].unique():
        return f"예측된 클러스터 라벨 {predicted_label}는 유효하지 않습니다."

    cluster_data = model_data[model_data['ClusterLabel'] == predicted_label]
    top_addresses_and_products = cluster_data.groupby(['address', 'product']).size().reset_index(name='counts')
    top_addresses_and_products = top_addresses_and_products.sort_values(by='counts', ascending=False).head(5)
    top_addresses_and_products.drop(columns=['counts'],inplace=True)
    return top_addresses_and_products

# 예시: 새로운 X 값 입력
new_X_value = [30, 5086, 176]  # 예시 값입니다. 실제 값을 입력해 주세요.
top_addresses = get_addresses_by_prediction(new_X_value)
print(f"예측된 클러스터에 속하는 상위 5개 도로명주소:\n{top_addresses}")


#시각화

import matplotlib.pyplot as plt

# 클러스터 시각화를 위한 2D 플롯
plt.figure(figsize=(10, 7))
plt.scatter(X_features_scaled[:, 0], X_features_scaled[:, 1], c=Y_labels, cmap='viridis', marker='o', edgecolor='k')
plt.title('2D Visualization of Clusters')
plt.xlabel('Employees')
plt.ylabel('Area')
plt.show()




# 랜덤 포레스트 특성 중요도 시각화
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 7))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np

# 실루엣 점수 계산
silhouette_vals = silhouette_samples(X_features_scaled, Y_labels)

# 실루엣 플롯
y_ticks = []
y_lower, y_upper = 0, 0
plt.figure(figsize=(10, 7))
for i, cluster in enumerate(np.unique(Y_labels)):
    cluster_silhouette_vals = silhouette_vals[Y_labels == cluster]
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1.0)
    y_ticks.append((y_lower + y_upper) / 2)
    y_lower += len(cluster_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")
plt.yticks(y_ticks, np.unique(Y_labels) + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.title('Silhouette plot for the various clusters')
plt.show()



# 예측된 클러스터와 실제 클러스터를 비교하는 시각화

X_test_df = pd.DataFrame(X_test, columns=['Employees', 'Area', 'Power unit price'])
plt.figure(figsize=(10, 7))
plt.scatter(X_test['Employees'], X_test['Area'], c=Y_test, marker='o', edgecolor='k', label='Actual')
plt.scatter(X_test['Employees'], X_test['Area'], c=Y_predict, marker='x', edgecolor='r', label='Predicted')
plt.title('Actual vs Predicted Clusters')
plt.xlabel('Employees')
plt.ylabel('Area')
plt.legend()
plt.show()



















