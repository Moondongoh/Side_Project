# Week 3: 객체 탐지 모델 추론 및 시각화
# 이 코드는 YOLOv8 모델을 사용하여 객체 탐지를 수행하고 결과를 시각화하는 코드입니다.

import cv2
from ultralytics import YOLO

# 1. 학습된 YOLOv8 모델 로드
model = YOLO("C:/GIT/runs/cat-dog-person/weights/best.pt")

# 2. 테스트할 이미지 불러오기
image_path = "E:/cat-dog-person.v4i.yolov8/test/images/cat_217_jpg.rf.31c07bc572a385de34f50bd1325ac90a.jpg"
image = cv2.imread(image_path)

# 3. 객체 탐지 실행
results = model(image)[0]  # 리스트 중 첫 번째 결과 사용

# 4. 탐지된 객체 시각화
for box in results.boxes:
    # (1) 바운딩 박스 좌표 추출
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # (2) 클래스 라벨 및 신뢰도 추출
    cls_id = int(box.cls[0])
    label = results.names[cls_id]
    confidence = float(box.conf[0])

    # (3) 바운딩 박스 그리기
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"{label} ({confidence:.2f})",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

# 5. 결과 출력
cv2.imshow("YOLOv8 Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
