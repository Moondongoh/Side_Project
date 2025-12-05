# import numpy as np
# from PIL import Image


# def is_too_dark(image, brightness_threshold=20):
#     # Grayscale로 변환 후 numpy array
#     gray = image.convert("L")
#     np_gray = np.array(gray)
#     mean_brightness = np.mean(np_gray)
#     return mean_brightness < brightness_threshold


# def is_too_small(image, min_width=100, min_height=100):
#     w, h = image.size
#     return w < min_width or h < min_height


# def filter_images(example):
#     image = example["image"]
#     # 조건: 너무 어둡거나 너무 작으면 제거
#     if is_too_dark(image) or is_too_small(image):
#         return False
#     return True


# # Food-101 데이터셋에서 1000개만 로드
# from datasets import load_dataset

# dataset = load_dataset("food101", split="train[:1000]")

# # filter 적용
# filtered_dataset = dataset.filter(filter_images)

# print(f"원본 데이터 개수: {len(dataset)}")
# print(f"필터링 후 데이터 개수: {len(filtered_dataset)}")
# # 시각화 함수
# import matplotlib.pyplot as plt


# def show_images(images, titles=None):
#     plt.figure(figsize=(20, 5))
#     for i, img in enumerate(images):
#         plt.subplot(1, len(images), i + 1)
#         if isinstance(img, Image.Image):
#             plt.imshow(img)
#         else:
#             plt.imshow(img.permute(1, 2, 0).numpy())
#         if titles:
#             plt.title(titles[i])
#         plt.axis("off")
#     plt.show()


# # 시각화
# example_images = [filtered_dataset[i]["image"] for i in range(5)]
# show_images(example_images, titles=[f"Image {i+1}" for i in range(5)])
# # 시각화된 이미지 출력
# plt.show()


# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# from datasets import load_dataset


# # ======================
# # 필터링 조건 정의
# # ======================
# def is_too_dark(image, brightness_threshold=30):
#     gray = image.convert("L")
#     np_gray = np.array(gray)
#     mean_brightness = np.mean(np_gray)
#     return mean_brightness < brightness_threshold


# # 필터 함수 정의
# def filter_images(example):
#     image = example["image"]
#     if is_too_dark(image):
#         return False
#     return True


# # ======================
# # 데이터 로드 및 필터링
# # ======================
# dataset = load_dataset("food101", split="train[:1000]")

# filtered_dataset = dataset.filter(filter_images)

# print(f"원본 데이터 개수: {len(dataset)}")
# print(f"필터링 후 데이터 개수: {len(filtered_dataset)}")

# # ======================
# # 제거된 이미지 추출
# # ======================
# removed_items = [example for example in dataset if not filter_images(example)]
# print(f"필터링으로 제거된 데이터 개수: {len(removed_items)}")


# # ======================
# # 시각화 함수 정의
# # ======================
# def show_images(images, titles=None):
#     plt.figure(figsize=(20, 5))
#     for i, img in enumerate(images):
#         plt.subplot(1, len(images), i + 1)
#         if isinstance(img, Image.Image):
#             plt.imshow(img)
#         else:
#             plt.imshow(img.permute(1, 2, 0).numpy())
#         if titles:
#             plt.title(titles[i])
#         plt.axis("off")
#     plt.show()


# # ======================
# # 필터링된 예시 이미지 시각화
# # ======================
# example_images = [filtered_dataset[i]["image"] for i in range(5)]
# show_images(example_images, titles=[f"Filtered {i+1}" for i in range(5)])

# # ======================
# # 제거된 이미지 시각화 및 밝기 출력
# # ======================
# removed_images = [item["image"] for item in removed_items[:5]]
# show_images(
#     removed_images, titles=[f"Removed {i+1}" for i in range(len(removed_images))]
# )

# for i, item in enumerate(removed_items[:5]):
#     gray = item["image"].convert("L")
#     mean_brightness = np.mean(np.array(gray))
#     print(f"{i+1}번째 제거된 이미지의 평균 밝기: {mean_brightness:.2f}")

# ===============================================================================================================================
# Week 1: 고급 이미지 전처리 및 필터링
# 이 코드에서는 임계치 미만 밝기의 사진과 객체크기가 작은 이미지를 필터링하는 코드입니다.
# ver.2

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from datasets import load_dataset


# ======================
# 너무 어둡거나 너무 작은 이미지 필터링 함수
# ======================
def is_too_dark(image, brightness_threshold=30):
    gray = image.convert("L")
    np_gray = np.array(gray)
    mean_brightness = np.mean(np_gray)
    return mean_brightness < brightness_threshold


def is_too_small_object(image, min_object_area=100):
    """
    PIL 이미지 입력 → OpenCV Canny + contour로 객체별 크기 판단해서
    작은 객체가 있으면 False (필터링 대상) 반환
    """
    # PIL → OpenCV 변환 (RGB→BGR)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    median_intensity = np.median(gray)
    lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
    upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))

    edges = cv2.Canny(gray, lower_threshold, upper_threshold)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 최소 면적보다 큰 객체가 하나라도 있으면 True (충분히 큰 객체 있음)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_object_area:
            return False  # 너무 작은 객체만 있는게 아니므로 필터하지 않음

    # 모든 객체가 너무 작거나 객체 없음 → 필터링 대상
    return True


# ======================
# 데이터셋 필터링 함수 통합
# ======================
def filter_images(example):
    image = example["image"]

    if is_too_dark(image):
        return False  # 너무 어두움 → 필터링
    if is_too_small_object(image):
        return False  # 너무 작은 객체만 있음 → 필터링
    return True


# ======================
# 데이터 로드 및 필터링
# ======================
dataset = load_dataset("food101", split="train[:1000]")

filtered_dataset = dataset.filter(filter_images)

print(f"원본 데이터 개수: {len(dataset)}")
print(f"필터링 후 데이터 개수: {len(filtered_dataset)}")

# ======================
# 제거된 이미지 추출 (필터링 False 된 항목)
# ======================
removed_items = [example for example in dataset if not filter_images(example)]
print(f"필터링으로 제거된 데이터 개수: {len(removed_items)}")


# ======================
# 시각화 함수
# ======================
def show_images(images, titles=None):
    plt.figure(figsize=(20, 5))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        if isinstance(img, Image.Image):
            plt.imshow(img)
        else:
            plt.imshow(img.permute(1, 2, 0).numpy())
        if titles:
            plt.title(titles[i])
        plt.axis("off")
    plt.show()


# ======================
# 필터링된 예시 이미지 시각화
# ======================
example_images = [filtered_dataset[i]["image"] for i in range(5)]
show_images(example_images, titles=[f"Filtered {i+1}" for i in range(5)])

# ======================
# 제거된 이미지 시각화 및 밝기 출력
# ======================
removed_images = [item["image"] for item in removed_items[:5]]
show_images(
    removed_images, titles=[f"Removed {i+1}" for i in range(len(removed_images))]
)

for i, item in enumerate(removed_items[:5]):
    gray = item["image"].convert("L")
    mean_brightness = np.mean(np.array(gray))
    print(f"{i+1}번째 제거된 이미지의 평균 밝기: {mean_brightness:.2f}")
