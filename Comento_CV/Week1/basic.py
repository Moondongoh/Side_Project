# Week 1: 이미지 전처리 및 시각화
# 이 코드는 OpenCV를 사용하여 다양한 이미지 증강을 해본 코드입니다.

import os
import torch
import numpy as np
from datasets import load_dataset
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

# 현재 파일 위치 기준으로 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "result")

# result 폴더 생성
os.makedirs(RESULT_DIR, exist_ok=True)

# 데이터셋 로드
dataset = load_dataset("food101", split="train[:5]")  # 5장 로드

# 전처리 변환
resize_transform = transforms.Resize((224, 224))
grayscale_normalize_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)
augment_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ]
)


# 저장 함수
def save_image(img, path):
    if isinstance(img, torch.Tensor):
        img = img.detach()
        img = img.permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img.squeeze())
    img.save(path)


# 시각화 함수
def show_images(images, titles=None):
    plt.figure(figsize=(20, 5))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        if isinstance(img, torch.Tensor):
            img = img.detach()
            img = img.permute(1, 2, 0).numpy()
            img = (img * 0.5 + 0.5).clip(0, 1)
            plt.imshow(img.squeeze(), cmap="gray")
        else:
            plt.imshow(img, cmap="gray")
        if titles:
            plt.title(titles[i])
        plt.axis("off")
    plt.show()


# 이미지 처리
for idx, example in enumerate(dataset):
    original = example["image"]
    resized = resize_transform(original)
    grayscaled_tensor = grayscale_normalize_transform(resized)
    blurred = resized.filter(ImageFilter.GaussianBlur(radius=1))
    augmented = augment_transform(resized)

    save_image(original, os.path.join(RESULT_DIR, f"{idx}_original.png"))
    save_image(resized, os.path.join(RESULT_DIR, f"{idx}_resize.png"))
    save_image(grayscaled_tensor, os.path.join(RESULT_DIR, f"{idx}_grayscale.png"))
    save_image(blurred, os.path.join(RESULT_DIR, f"{idx}_blur.png"))
    save_image(augmented, os.path.join(RESULT_DIR, f"{idx}_augment.png"))

    show_images(
        [original, resized, grayscaled_tensor, blurred, augmented],
        ["Original", "Resize(224x224)", "Grayscale+Normalize", "Blur", "Augmentation"],
    )

    print(f"Saved processed images for sample {idx}")

print("모든 이미지 저장 및 시각화 완료")
