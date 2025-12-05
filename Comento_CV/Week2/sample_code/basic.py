# Week 2: OpenCV를 이용한 가상의 깊이맵 생성

import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치
PARENT_DIR = os.path.dirname(BASE_DIR)  # 상위 디렉토리
image_path = os.path.join(PARENT_DIR, "input_image", "images.png")

# 이미지 로드
image = cv2.imread(image_path)

# 이미지가 제대로 로드되었는지 확인
if image is None:
    raise FileNotFoundError(
        "이미지를 찾을 수 없습니다. 'sample.jpg' 경로를 확인하세요."
    )

# 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 깊이맵 생성 (가상의 깊이 적용)
depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

# 결과 출력
cv2.imshow("Original Image", image)
cv2.imshow("DepthMap", depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
