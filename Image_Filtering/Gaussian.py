"""
* 2 가우시안 필터링
-1. 그레이 영상의 외곽을 제외한 모든 픽셀에 마스크를 씌운다.
-2. 마스크를 씌운 값을 마스크의 총 합 으로 나눠준다.
-3. 나누어진 값을 입력한다.
"""

import numpy as np


def gaussian_kernel(size=3, sigma=1.0):
    assert size % 2 == 1, "커널 크기는 홀수여야 합니다."

    k = size // 2
    x, y = np.mgrid[-k : k + 1, -k : k + 1]

    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)  # 정규화 (총합 = 1)

    return kernel


def apply_gaussian_filter(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad = kh // 2

    output = np.zeros_like(image, dtype=np.float32)

    kernel_sum = np.sum(kernel)  # 보통 1

    for y in range(pad, h - pad):
        for x in range(pad, w - pad):
            region = image[y - pad : y + pad + 1, x - pad : x + pad + 1]
            output[y, x] = np.sum(region * kernel) / kernel_sum

    return output


import cv2

img = cv2.imread(r"C:\Users\Moon\Desktop\filter.jpg", cv2.IMREAD_GRAYSCALE)

kernel = gaussian_kernel(size=5, sigma=1.0)

gaussian_result = apply_gaussian_filter(img, kernel)

gaussian_result = np.uint8(np.clip(gaussian_result, 0, 255))

cv2.imshow("Gaussian Filter Result", gaussian_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
