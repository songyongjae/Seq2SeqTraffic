import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import shutil
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# MPS 장치 설정
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# 사전 훈련된 ResNet 모델 로드 및 중간 레이어 출력 모델 생성
class FeatureExtractor(nn.Module):
    def __init__(self, submodule):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule

    def forward(self, x):
        layers = list(self.submodule.children())[:-1]  # 마지막 분류기 레이어 제거
        for layer in layers:
            x = layer(x)
        return x

resnet = models.resnet50(weights='IMAGENET1K_V1').to(device)
# 첫 번째 합성곱 레이어 변경
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

model = FeatureExtractor(resnet).to(device)
model.eval()

# 이미지 전처리
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

def extract_features(img_path, model, device):
    try:
        img = Image.open(img_path).convert('L')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(img_tensor)
        return features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

# 이미지 데이터로부터 특징 추출
image_dir = 'ProjectImages'  # 이미지 파일 경로
output_dir = 'clustered2'  # 분류된 이미지가 저장될 디렉토리
os.makedirs(output_dir, exist_ok=True)

features = []
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

for img_file in image_files:
    feature = extract_features(img_file, model, device)
    if feature is not None:
        features.append(feature)

features = np.array(features)
df1 = pd.DataFrame(features)
df1.to_excel('features2.xlsx', index=False)

if features.size == 0:
    raise ValueError("No features extracted. Please check your image files and preprocessing steps.")

# 차원 축소 (선택 사항)
pca = PCA(n_components=50)
features_reduced = pca.fit_transform(features)
df2 = pd.DataFrame(features_reduced)
df2.to_excel('pca2.xlsx', index=False)

# Elbow 방식으로 최적의 k 찾기
wcss = []
max_k = 40
for k in range(1, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_reduced)
    wcss.append(kmeans.inertia_)

'''
# Elbow 포인트 찾기 (1차 미분 사용)
diffs = np.diff(wcss)
second_diffs = np.diff(diffs)
optimal_k = np.argmax(second_diffs) + 2  # 1차 미분의 최대값의 인덱스 + 2
'''

# Elbow 포인트 찾기 (기울기 변화 사용)
diffs = np.diff(wcss)
optimal_k = np.argmax(diffs[:-1] - diffs[1:]) + 2

# Elbow 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
plt.axvline(x=optimal_k, linestyle='--', color='r', label='Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal k')
plt.legend()
plt.savefig('elbow2.png')
plt.show()

# 최적의 k를 사용하여 K-means 클러스터링 수행
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(features_reduced)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 이상치 감지 임계값 설정 (예: 평균 거리의 1.3배 이상인 경우)
threshold_factor = 1.3
distances = np.linalg.norm(features_reduced - centroids[labels], axis=1)
threshold = np.mean(distances) * threshold_factor

# 이상치 감지
outliers = distances > threshold
outlier_indices = np.where(outliers)[0]
normal_indices = np.where(~outliers)[0]

# 각 군집별 폴더 생성 및 이미지 복사
for cluster_num in range(optimal_k):
    cluster_dir = os.path.join(output_dir, f'cluster_{cluster_num}')
    os.makedirs(cluster_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'outliers'), exist_ok=True)

# 이미지를 해당 군집 또는 이상치 폴더로 복사
for idx, img_file in enumerate(image_files):
    if idx in outlier_indices:
        cluster_dir = os.path.join(output_dir, 'outliers')
    else:
        cluster_num = labels[idx]
        cluster_dir = os.path.join(output_dir, f'cluster_{cluster_num}')
    shutil.copy(img_file, cluster_dir)
    print(f"Image {img_file} is copied to {cluster_dir}")

print("Image clustering and saving completed.")

# 각 군집의 표준편차 계산
cluster_std = []
for cluster_num in range(optimal_k):
    cluster_features = features_reduced[labels == cluster_num]
    cluster_std.append(np.std(cluster_features))

'''
# 표준편차를 정규화하여 가중치 계산 (0.85 ~ 1.28)
scaler = MinMaxScaler(feature_range=(0.85, 1.28))
cluster_std = scaler.fit_transform(np.array(cluster_std).reshape(-1, 1)).flatten()
'''

# 각 군집별 가중치 출력
for cluster_num, weight in enumerate(cluster_std):
    print(f"Cluster {cluster_num} weight: {weight}")

# 군집별 가중치 저장 (선택 사항)
df_weights = pd.DataFrame({'Cluster': range(optimal_k), 'Weight': cluster_std})
df_weights.to_excel('cluster_weights2.xlsx', index=False)