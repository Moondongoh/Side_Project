# week4 Image Processing
# OpenCV를 사용하여 이미지에서 차선을 감지하는 코드입니다.

import cv2 as cv
import numpy as np


def lane_detection(image_path):
    img = cv.imread(image_path)
    if img is None:
        print("이미지를 불러올 수 없습니다.")
        return

    height = img.shape[0]  # 세로 크기
    width = img.shape[1]  # 가로 크기

    # ROI 설정 (아래쪽 1/3 영역만)
    roi_start_y = int(height * 2 / 3)
    ROI = img[roi_start_y:height, :]

    # 그레이스케일 및 블러
    gray_img = cv.cvtColor(ROI, cv.COLOR_BGR2GRAY)
    blur_img = cv.GaussianBlur(gray_img, (5, 5), 0)

    # Canny 에지 검출
    canny_img = cv.Canny(blur_img, 50, 150)

    # 허프 변환으로 직선 검출
    linesP = cv.HoughLinesP(
        canny_img, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=5
    )

    if linesP is not None:
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            cv.line(
                img,
                (x1, y1 + roi_start_y),
                (x2, y2 + roi_start_y),
                (0, 0, 255),
                3,
                cv.LINE_AA,
            )

    cv.imshow("Lane Detection", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    lane_detection("./image/line.png")
