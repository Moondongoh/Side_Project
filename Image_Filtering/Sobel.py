"""
* 1 소벨 에지 검출
-1. 그레이 영상의 외곽을 제외한 모든 픽셀에 수직 마스크와 수평 마스크를 씌운다.
-2. 구해진 두 값의 절대값 끼리 더한다.
-3. 영상 히스토그램의 평균값과 비교하여 클 경우 255값을 작을 경우 0 값을 입력한다.
"""

import cv2
import numpy as np

img = cv2.imread(r"C:\Users\Moon\Desktop\filter.jpg", cv2.IMREAD_GRAYSCALE)

# dx=1, dy=0 → 수직 방향 엣지 (Vertical mask)
grad_x = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=3)

# dx=0, dy=1 → 수평 방향 엣지 (Horizontal mask)
grad_y = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=3)

grad_x = cv2.convertScaleAbs(grad_x)
grad_y = cv2.convertScaleAbs(grad_y)

cv2.imshow("Vertical Mask Result", grad_x)
cv2.imshow("Horizontal Mask Result", grad_y)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Numpy를 사용한 수동 구현 예시
# import cv2
# import numpy as np

# def convolution(image, kernel):
#     h, w = image.shape
#     kh, kw = kernel.shape
#     pad = kh // 2

#     padded = np.pad(image, pad, mode='constant')
#     output = np.zeros_like(image, dtype=np.float32)

#     for y in range(h):
#         for x in range(w):
#             region = padded[y:y+kh, x:x+kw]
#             output[y, x] = np.sum(region * kernel)

#     return output

# # 이미지 로드
# img = cv2.imread(r"C:\Users\Moon\Desktop\filter.jpg", cv2.IMREAD_GRAYSCALE)

# # Sobel 마스크 정의
# vertical_mask = np.array([
#     [-1, 0, 1],
#     [-2, 0, 2],
#     [-1, 0, 1]
# ])

# horizontal_mask = np.array([
#     [-1, -2, -1],
#     [ 0,  0,  0],
#     [ 1,  2,  1]
# ])

# # 마스크 적용
# vertical_result = convolution(img, vertical_mask)
# horizontal_result = convolution(img, horizontal_mask)

# # 결과 정규화
# vertical_result = np.uint8(np.clip(np.abs(vertical_result), 0, 255))
# horizontal_result = np.uint8(np.clip(np.abs(horizontal_result), 0, 255))

# cv2.imshow("Vertical Mask Result", vertical_result)
# cv2.imshow("Horizontal Mask Result", horizontal_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
