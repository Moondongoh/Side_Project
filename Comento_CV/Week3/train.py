# Week 3: YOLOv8 모델 학습
# 이 코드는 YOLOv8 모델을 학습시키는 코드입니다.

from ultralytics import YOLO
import torch


def main():
    print(torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # 모델 로드
    model = YOLO("yolov8n.pt")

    # 학습 실행
    model.train(
        data="C:/Users/as/Desktop/dong/cat-dog-person.v4i.yolov8/data.yaml",  # roboflow 데이터셋 다운로드
        epochs=20,
        imgsz=640,
        batch=8,
        device="0",
        project="runs",
        name="cat-dog-person",
        # 데이터 증강 하이퍼파라미터
        hsv_h=0.015,  # 색조 변화
        hsv_s=0.7,  # 채도 변화
        hsv_v=0.4,  # 명도 변화
        degrees=10.0,  # 회전
        translate=0.1,  # 이동
        scale=0.5,  # 확대/축소
        shear=2.0,  # 기울이기
        flipud=0.0,  # 상하 뒤집기 확률
        fliplr=0.5,  # 좌우 뒤집기 확률
        mosaic=1.0,  # 모자이크 사용
        mixup=0.2,  # MixUp 사용
        copy_paste=0.0,  # 복붙 증강 (Copy-Paste)
    )


# 이걸로 실행
if __name__ == "__main__":
    main()
