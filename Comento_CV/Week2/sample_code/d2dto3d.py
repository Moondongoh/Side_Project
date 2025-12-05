# Week 2: 2D 이미지에서 3D 포인트 클라우드 생성
# 이 코드는 OpenCV와 Matplotlib를 사용하여 2D 이미지를 3D 포인트 클라우드로 변환하고 시각화하는 코드입니다.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치
PARENT_DIR = os.path.dirname(BASE_DIR)  # 상위 디렉토리
image_path = os.path.join(PARENT_DIR, "input_image", "images.png")

# 이미지 로드
image = cv2.imread(image_path)

image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("이미지를 불러올 수 없습니다. 경로를 확인하세요.")

# Grayscale → DepthMap 생성
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

# 3D 포인트 생성
h, w = gray.shape
X, Y = np.meshgrid(np.arange(w), np.arange(h))
Z = gray.astype(np.float32)

points = np.dstack((X, Y, Z)).reshape(-1, 3)  # (N, 3)
colors = image.reshape(-1, 3) / 255.0  # RGB (0~1 정규화)

# 포인트 일부 샘플링 (너무 많으면 렌더링 느려짐)
num_samples = 5000
indices = np.random.choice(len(points), size=num_samples, replace=False)
sampled_points = points[indices]
sampled_colors = colors[indices]

# 3D 시각화
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    sampled_points[:, 0],  # X
    sampled_points[:, 1],  # Y
    sampled_points[:, 2],  # Z
    c=sampled_colors,
    s=0.5,
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Depth (Z)")
ax.set_title("3D Point Cloud (Simulated Depth)")
plt.tight_layout()
plt.show()


def compute_3d_points_from_image(image, num_samples=5000):
    if image is None:
        raise ValueError("이미지가 없습니다.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray.astype(np.float32)

    points = np.dstack((X, Y, Z)).reshape(-1, 3)  # (N, 3)
    colors = image.reshape(-1, 3) / 255.0

    if len(points) < num_samples:
        raise ValueError("샘플링할 수 있는 포인트가 부족합니다.")

    indices = np.random.choice(len(points), size=num_samples, replace=False)
    return points[indices], colors[indices]
