# Week 2: OpenCV를 이용한 가상의 깊이맵 생성

import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치
PARENT_DIR = os.path.dirname(BASE_DIR)  # 상위 디렉토리
image_path = os.path.join(PARENT_DIR, "input_image", "images.png")

# 이미지 로드
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("이미지를 불러올 수 없습니다.")

# 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Depth Map 생성 (가짜 깊이)
depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

# 3D 포인트 클라우드 생성
h, w = gray.shape
X, Y = np.meshgrid(np.arange(w), np.arange(h))
Z = gray.astype(np.float32)  # 깊이 정보로 사용

# 3D 좌표 결합
points_3d = np.dstack((X, Y, Z))  # shape: (h, w, 3)

# 결과 출력 (2D 컬러맵만 보여줌)
cv2.imshow("DepthMap", depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
