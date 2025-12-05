import cv2 as cv
import numpy as np
import logging
from datetime import datetime
import json
import os
from typing import Dict, List, Tuple, Optional, Union
import base64
from io import BytesIO
from PIL import Image

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/tmp/red_line_detection.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class RedLineDetector:
    def __init__(self, config: Optional[Dict] = None):
        """
        빨간색 선 검출기 초기화

        Args:
            config: 설정 딕셔너리
                - red_lower1/red_upper1: 첫 번째 빨간색 HSV 범위
                - red_lower2/red_upper2: 두 번째 빨간색 HSV 범위
                - roi_start_ratio: ROI 시작 비율 (0.5 = 화면 아래쪽 절반)
                - min_area: 최소 면적 임계값
                - tolerance: 중앙선 허용 오차
        """
        # 기본 설정
        default_config = {
            "red_lower1": [0, 50, 50],
            "red_upper1": [10, 255, 255],
            "red_lower2": [170, 50, 50],
            "red_upper2": [180, 255, 255],
            "roi_start_ratio": 0.5,
            "min_area": 500,
            "tolerance": 25,
            "kernel_size": 5,
            "blur_size": 5,
        }

        self.config = {**default_config, **(config or {})}

        # HSV 범위 설정
        self.red_lower1 = np.array(self.config["red_lower1"])
        self.red_upper1 = np.array(self.config["red_upper1"])
        self.red_lower2 = np.array(self.config["red_lower2"])
        self.red_upper2 = np.array(self.config["red_upper2"])

        logger.info("빨간색 선 검출기 초기화 완료")
        logger.info(f"설정: {self.config}")

    def load_image(
        self, input_data: Union[str, np.ndarray, bytes]
    ) -> Optional[np.ndarray]:
        """
        다양한 형식의 입력 데이터를 이미지로 로드

        Args:
            input_data: 이미지 파일 경로, numpy 배열, 또는 바이트 데이터

        Returns:
            np.ndarray: OpenCV 이미지 또는 None
        """
        try:
            if isinstance(input_data, str):
                # 파일 경로인 경우
                if os.path.exists(input_data):
                    img = cv.imread(input_data)
                    logger.info(f"이미지 파일 로드: {input_data}")
                    return img
                # Base64 문자열인 경우
                elif input_data.startswith("data:image"):
                    # data:image/jpeg;base64,... 형식 처리
                    header, data = input_data.split(",", 1)
                    img_bytes = base64.b64decode(data)
                else:
                    # 일반 base64 문자열
                    img_bytes = base64.b64decode(input_data)

                # 바이트에서 이미지 로드
                if "img_bytes" in locals():
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
                    logger.info("Base64 이미지 로드 완료")
                    return img

            elif isinstance(input_data, bytes):
                # 바이트 데이터인 경우
                nparr = np.frombuffer(input_data, np.uint8)
                img = cv.imdecode(nparr, cv.IMREAD_COLOR)
                logger.info("바이트 이미지 로드 완료")
                return img

            elif isinstance(input_data, np.ndarray):
                # 이미 numpy 배열인 경우
                logger.info("NumPy 배열 이미지 사용")
                return input_data.copy()

        except Exception as e:
            logger.error(f"이미지 로드 실패: {e}")

        return None

    def detect_red_line(self, img: np.ndarray) -> Dict:
        """
        이미지에서 빨간색 선 검출

        Args:
            img: OpenCV 이미지 (BGR)

        Returns:
            Dict: 검출 결과
                - line_detected: bool, 선 검출 여부
                - center_x: float, 선의 중심 X 좌표
                - center_y: float, 선의 중심 Y 좌표
                - area: float, 검출된 영역의 면적
                - contours: list, 검출된 윤곽선들
                - confidence: float, 검출 신뢰도 (0-1)
                - direction: str, 권장 이동 방향
                - error: float, 중앙선 대비 오차
        """
        if img is None:
            return self._empty_result()

        height, width = img.shape[:2]

        # ROI 설정 (화면 아래쪽)
        roi_start_y = int(height * self.config["roi_start_ratio"])
        roi = img[roi_start_y:height, :]
        roi_height = roi.shape[0]

        # HSV 변환
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

        # 빨간색 마스크 생성 (두 범위를 합침)
        mask1 = cv.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv.bitwise_or(mask1, mask2)

        # 노이즈 제거
        kernel_size = self.config["kernel_size"]
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)
        red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)

        # 가우시안 블러로 부드럽게
        blur_size = self.config["blur_size"]
        red_mask = cv.GaussianBlur(red_mask, (blur_size, blur_size), 0)

        # 윤곽선 찾기
        contours, _ = cv.findContours(
            red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        # 결과 초기화
        result = {
            "line_detected": False,
            "center_x": width / 2,
            "center_y": roi_start_y + roi_height / 2,
            "area": 0,
            "contours": [],
            "confidence": 0.0,
            "direction": "stop",
            "error": 0.0,
            "mask": red_mask,
            "roi_bounds": (0, roi_start_y, width, height),
        }

        if contours:
            # 가장 큰 윤곽선 선택
            largest_contour = max(contours, key=cv.contourArea)
            area = cv.contourArea(largest_contour)

            if area > self.config["min_area"]:
                # 무게중심 계산
                moments = cv.moments(red_mask)
                if moments["m00"] > 0:
                    center_x = moments["m10"] / moments["m00"]
                    center_y = roi_start_y + (moments["m01"] / moments["m00"])

                    # 신뢰도 계산 (면적 기반)
                    max_possible_area = width * roi_height
                    confidence = min(
                        area / (max_possible_area * 0.1), 1.0
                    )  # 최대 10% 기준

                    # 방향 및 오차 계산
                    error = center_x - (width / 2)
                    direction = self._calculate_direction(error)

                    result.update(
                        {
                            "line_detected": True,
                            "center_x": center_x,
                            "center_y": center_y,
                            "area": area,
                            "contours": [
                                largest_contour.tolist()
                            ],  # JSON 직렬화 가능하게
                            "confidence": confidence,
                            "direction": direction,
                            "error": error,
                        }
                    )

        return result

    def _calculate_direction(self, error: float) -> str:
        """오차를 바탕으로 이동 방향 계산"""
        tolerance = self.config["tolerance"]

        if abs(error) < tolerance:
            return "forward"
        elif error < -tolerance:
            if abs(error) > 80:
                return "sharp_left"
            else:
                return "left"
        else:
            if abs(error) > 80:
                return "sharp_right"
            else:
                return "right"

    def _empty_result(self) -> Dict:
        """빈 결과 반환"""
        return {
            "line_detected": False,
            "center_x": 0,
            "center_y": 0,
            "area": 0,
            "contours": [],
            "confidence": 0.0,
            "direction": "stop",
            "error": 0.0,
            "mask": None,
            "roi_bounds": (0, 0, 0, 0),
        }

    def visualize_result(self, img: np.ndarray, result: Dict) -> np.ndarray:
        """
        검출 결과를 이미지에 시각화

        Args:
            img: 원본 이미지
            result: detect_red_line의 결과

        Returns:
            np.ndarray: 시각화된 이미지
        """
        if img is None:
            return None

        vis_img = img.copy()
        height, width = img.shape[:2]

        # ROI 영역 표시
        roi_bounds = result["roi_bounds"]
        cv.rectangle(
            vis_img,
            (roi_bounds[0], roi_bounds[1]),
            (roi_bounds[2], roi_bounds[3]),
            (100, 100, 100),
            2,
        )
        cv.putText(
            vis_img,
            "ROI",
            (10, roi_bounds[1] - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (100, 100, 100),
            2,
        )

        # 검출된 선이 있는 경우
        if result["line_detected"]:
            # 윤곽선 그리기
            if result["contours"]:
                contours = [
                    np.array(contour, dtype=np.int32) for contour in result["contours"]
                ]
                # ROI 오프셋 적용
                offset_contours = []
                for contour in contours:
                    offset_contour = contour.copy()
                    offset_contour[:, :, 1] += roi_bounds[1]  # Y 좌표에 ROI 오프셋 추가
                    offset_contours.append(offset_contour)
                cv.drawContours(vis_img, offset_contours, -1, (0, 255, 0), 3)

            # 중심점 표시
            center_x, center_y = int(result["center_x"]), int(result["center_y"])
            cv.circle(vis_img, (center_x, center_y), 10, (0, 255, 255), -1)
            cv.putText(
                vis_img,
                f"CENTER",
                (center_x - 30, center_y - 15),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

            # 중앙선까지 연결선
            cv.line(
                vis_img, (width // 2, center_y), (center_x, center_y), (255, 0, 255), 2
            )

            status_color = (0, 255, 0)
        else:
            status_color = (0, 0, 255)
            cv.putText(
                vis_img,
                "RED LINE NOT FOUND!",
                (50, 100),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                status_color,
                3,
            )

        # 상태 정보 텍스트
        texts = [
            f"Detected: {result['line_detected']}",
            f"Direction: {result['direction']}",
            f"Error: {result['error']:.1f}px",
            f"Area: {result['area']:.0f}",
            f"Confidence: {result['confidence']:.2f}",
            f"Center: ({result['center_x']:.1f}, {result['center_y']:.1f})",
        ]

        for i, text in enumerate(texts):
            y_pos = 30 + i * 25
            cv.putText(
                vis_img,
                text,
                (10, y_pos),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
            )

        # 화면 중앙선 표시
        cv.line(vis_img, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)

        # 허용 범위 표시
        tolerance = self.config["tolerance"]
        left_bound = width // 2 - tolerance
        right_bound = width // 2 + tolerance
        cv.line(
            vis_img,
            (left_bound, height - 50),
            (right_bound, height - 50),
            (0, 255, 0),
            5,
        )
        cv.putText(
            vis_img,
            "OK ZONE",
            (left_bound, height - 60),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        return vis_img

    def process_image(
        self,
        input_data: Union[str, np.ndarray, bytes],
        visualize: bool = True,
        save_result: bool = False,
        output_path: str = None,
    ) -> Dict:
        """
        이미지 처리 메인 함수

        Args:
            input_data: 입력 이미지 (경로, numpy 배열, 또는 바이트)
            visualize: 시각화 이미지 생성 여부
            save_result: 결과 저장 여부
            output_path: 출력 경로 (None이면 자동 생성)

        Returns:
            Dict: 전체 처리 결과
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 이미지 로드
        img = self.load_image(input_data)
        if img is None:
            logger.error("이미지를 로드할 수 없습니다")
            return {"success": False, "error": "Failed to load image"}

        # 빨간색 선 검출
        detection_result = self.detect_red_line(img)

        result = {
            "success": True,
            "timestamp": timestamp,
            "input_shape": img.shape,
            "detection": detection_result,
        }

        # 시각화
        if visualize:
            vis_img = self.visualize_result(img, detection_result)
            result["visualization"] = vis_img

            # 시각화 이미지 저장
            if save_result:
                if output_path is None:
                    output_path = f"/tmp/red_line_result_{timestamp}.jpg"
                cv.imwrite(output_path, vis_img)
                result["output_path"] = output_path
                logger.info(f"결과 이미지 저장: {output_path}")

        # JSON 결과 저장
        if save_result:
            json_path = f"/tmp/red_line_result_{timestamp}.json"
            # 시각화 이미지와 마스크는 JSON에서 제외
            json_result = {
                k: v
                for k, v in result.items()
                if k not in ["visualization", "detection"]
            }
            json_result["detection"] = {
                k: v for k, v in detection_result.items() if k not in ["mask"]
            }

            with open(json_path, "w") as f:
                json.dump(json_result, f, indent=2, default=str)
            result["json_path"] = json_path
            logger.info(f"결과 JSON 저장: {json_path}")

        return result

    def batch_process(
        self,
        input_list: List[Union[str, np.ndarray, bytes]],
        visualize: bool = True,
        save_results: bool = False,
    ) -> List[Dict]:
        """
        여러 이미지 일괄 처리

        Args:
            input_list: 입력 이미지 리스트
            visualize: 시각화 여부
            save_results: 결과 저장 여부

        Returns:
            List[Dict]: 각 이미지의 처리 결과 리스트
        """
        results = []

        for i, input_data in enumerate(input_list):
            logger.info(f"처리 중: {i+1}/{len(input_list)}")

            try:
                result = self.process_image(
                    input_data, visualize=visualize, save_result=save_results
                )
                result["batch_index"] = i
                results.append(result)

            except Exception as e:
                logger.error(f"이미지 {i} 처리 실패: {e}")
                results.append({"success": False, "batch_index": i, "error": str(e)})

        return results


def main():
    """사용 예시"""
    import sys

    # 검출기 초기화
    detector = RedLineDetector()

    if len(sys.argv) < 2:
        print("사용법: python red_line_detector.py <이미지_경로> [visualize] [save]")
        print("예시: python red_line_detector.py test.jpg true true")
        return

    image_path = sys.argv[1]
    visualize = len(sys.argv) > 2 and sys.argv[2].lower() == "true"
    save_result = len(sys.argv) > 3 and sys.argv[3].lower() == "true"

    # 이미지 처리
    result = detector.process_image(
        image_path, visualize=visualize, save_result=save_result
    )

    if result["success"]:
        detection = result["detection"]
        print(f"\n=== 빨간색 선 검출 결과 ===")
        print(f"선 검출: {detection['line_detected']}")
        print(f"중심 좌표: ({detection['center_x']:.1f}, {detection['center_y']:.1f})")
        print(f"면적: {detection['area']:.0f}")
        print(f"신뢰도: {detection['confidence']:.2f}")
        print(f"권장 방향: {detection['direction']}")
        print(f"오차: {detection['error']:.1f}px")

        if save_result and "output_path" in result:
            print(f"결과 저장: {result['output_path']}")
    else:
        print(f"처리 실패: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
