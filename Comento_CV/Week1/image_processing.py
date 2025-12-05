# Week 1: 이미지 전처리 및 시각화
# 이 코드는 OpenCV를 사용하여 이미지에서 특정 색상을 검출하고 시각화하는 코드입니다.

import cv2
import numpy as np
import os

# # 이미지로드
# image = cv2.imread(
#     "C:/GIT/Comento_CV/test_processing/sample3.png"
# )  # 분석할 이미지 파일


# 현재 파일의 절대 경로 기반으로 이미지 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "test_processing", "sample3.png")

# 디버그: 경로 출력
print("[DEBUG] 이미지 경로:", image_path)
print("현재 파일 위치:", __file__)
print("실제 이미지 경로:", image_path)
print("이미지 존재 여부:", os.path.exists(image_path))

# 이미지 로드
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(
        f"[ERROR] 이미지가 존재하지 않거나 열 수 없습니다: {image_path}"
    )


# BGR에서HSV 색상공간으로변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 빨간색범위지정(두개의범위를설정해야함)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# 초록색 검출
lower_green = np.array([36, 100, 100])
upper_green = np.array([86, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)
result_green = cv2.bitwise_and(image, image, mask=mask_green)

# 파란색 검출
lower_blue = np.array([94, 80, 2])
upper_blue = np.array([126, 255, 255])
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
result_blue = cv2.bitwise_and(image, image, mask=mask_blue)

# 마스크생성
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = mask1 + mask2  # 두개의마스크를합침

# 원본이미지에서빨간색부분만추출
result = cv2.bitwise_and(image, image, mask=mask)

# 결과이미지출력
cv2.imshow("Original", image)
cv2.imshow("Red Filtered", result)
cv2.imshow("Green", result_green)
cv2.imshow("Blue", result_blue)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ✅실행결과: 빨간색영역이검출되며, 다른색상은제거된상태로표시됨
