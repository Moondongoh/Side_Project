# Week 3: YOLOv8 모델 학습 및 평가
# 이 코드는 YOLOv8 모델을 학습시키고, 평가 지표를 확인하는 코드입니다.

# from ultralytics import YOLO
# import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # 학습 완료된 모델 로드 (예: best.pt)
# model = YOLO("C:/GIT/runs/cat-dog-person/weights/best.pt")

# # val 데이터셋에 대해 평가
# metrics = model.val(data="E:/cat-dog-person.v4i.yolov8/data.yaml")

# # 지표 출력
# print(metrics.box)  # Precision, Recall, mAP 등
# print(metrics.speed)  # Inference 속도 등

# ============================================================================

# Week 3: YOLOv8 모델 학습 및 평가 시각화
# 이 코드는 전체 학습이 기록된 result.csv파일을(그래프로) 시각화하는 코드입니다.

import pandas as pd
import matplotlib.pyplot as plt

# 학습 로그 불러오기
log_path = "C:/GIT/runs/cat-dog-person/results.csv"
df = pd.read_csv(log_path)

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(df["metrics/precision(B)"], label="Precision")
plt.plot(df["metrics/recall(B)"], label="Recall")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Precision and Recall per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
