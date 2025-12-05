# week4
# HSV를 이용하여 입력 이미지에서 빨간색 선을 탐지하고 L9110 모터를 이용해 선을 따라 움직임을 구현하였습니다.

import cv2 as cv
import numpy as np
import time
import threading
import logging
from pathlib import Path
from datetime import datetime
import json
import os
import signal
import sys
from collections import deque

"""
RPi.GPIO 모듈은 라즈베리파이 전용이므로 Windows/테스트 환경에서 ImportError가 날 수 있다.
아래 try/except 블록이 mock 대체 모듈을 제공한다.
!!! 절대 이 블록보다 먼저 "import RPi.GPIO as GPIO"를 하지 말 것 !!!
"""
try:
    import RPi.GPIO as GPIO  # 실제 Pi 환경
except ImportError:
    # Windows용 Mock GPIO (라즈베리파이가 아닌 환경에서 테스트할 때)
    class GPIO:  # 최소 동작만 제공
        BCM = OUT = None

        @staticmethod
        def setmode(mode):
            pass

        @staticmethod
        def setup(pins, mode):
            pass

        @staticmethod
        def PWM(pin, freq):
            class PWM:
                def start(self, duty):
                    pass

                def ChangeDutyCycle(self, duty):
                    pass

                def stop(self):
                    pass

            return PWM()

        @staticmethod
        def cleanup():
            pass


# ----------------------------------------------------------------------
# 로깅 설정
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "red_line_following.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 모터 컨트롤러
# ----------------------------------------------------------------------
class MotorController:
    def __init__(
        self, left_pin1=18, left_pin2=19, right_pin1=20, right_pin2=21, pwm_freq=1000
    ):
        """
        L9110 모터 드라이버 초기화
        """
        self.left_pin1 = left_pin1
        self.left_pin2 = left_pin2
        self.right_pin1 = right_pin1
        self.right_pin2 = right_pin2

        # GPIO 설정
        GPIO.setmode(GPIO.BCM)
        GPIO.setup([left_pin1, left_pin2, right_pin1, right_pin2], GPIO.OUT)

        # PWM 설정
        self.left_pwm1 = GPIO.PWM(left_pin1, pwm_freq)
        self.left_pwm2 = GPIO.PWM(left_pin2, pwm_freq)
        self.right_pwm1 = GPIO.PWM(right_pin1, pwm_freq)
        self.right_pwm2 = GPIO.PWM(right_pin2, pwm_freq)

        # PWM 시작
        self.left_pwm1.start(0)
        self.left_pwm2.start(0)
        self.right_pwm1.start(0)
        self.right_pwm2.start(0)

        logger.info("모터 컨트롤러 초기화 완료")

    def move_forward(self, speed=60):
        """전진"""
        self.left_pwm1.ChangeDutyCycle(speed)
        self.left_pwm2.ChangeDutyCycle(0)
        self.right_pwm1.ChangeDutyCycle(speed)
        self.right_pwm2.ChangeDutyCycle(0)

    def turn_left(self, speed=50):
        """좌회전 (왼쪽 모터 느리게)"""
        self.left_pwm1.ChangeDutyCycle(speed * 0.4)
        self.left_pwm2.ChangeDutyCycle(0)
        self.right_pwm1.ChangeDutyCycle(speed)
        self.right_pwm2.ChangeDutyCycle(0)

    def turn_right(self, speed=50):
        """우회전 (오른쪽 모터 느리게)"""
        self.left_pwm1.ChangeDutyCycle(speed)
        self.left_pwm2.ChangeDutyCycle(0)
        self.right_pwm1.ChangeDutyCycle(speed * 0.4)
        self.right_pwm2.ChangeDutyCycle(0)

    def sharp_left(self, speed=50):
        """급좌회전"""
        self.left_pwm1.ChangeDutyCycle(0)
        self.left_pwm2.ChangeDutyCycle(0)
        self.right_pwm1.ChangeDutyCycle(speed)
        self.right_pwm2.ChangeDutyCycle(0)

    def sharp_right(self, speed=50):
        """급우회전"""
        self.left_pwm1.ChangeDutyCycle(speed)
        self.left_pwm2.ChangeDutyCycle(0)
        self.right_pwm1.ChangeDutyCycle(0)
        self.right_pwm2.ChangeDutyCycle(0)

    def stop(self):
        """정지"""
        self.left_pwm1.ChangeDutyCycle(0)
        self.left_pwm2.ChangeDutyCycle(0)
        self.right_pwm1.ChangeDutyCycle(0)
        self.right_pwm2.ChangeDutyCycle(0)

    def cleanup(self):
        """GPIO 정리"""
        self.stop()
        self.left_pwm1.stop()
        self.left_pwm2.stop()
        self.right_pwm1.stop()
        self.right_pwm2.stop()
        GPIO.cleanup()
        logger.info("모터 컨트롤러 정리 완료")


# ----------------------------------------------------------------------
# 웹 스트리머 (실기용)
# ----------------------------------------------------------------------
class WebStreamer:
    def __init__(self, port=8080):
        """웹 스트리밍을 위한 간단한 HTTP 서버"""
        self.port = port
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = False

    def update_frame(self, frame):
        """프레임 업데이트"""
        with self.frame_lock:
            _, buffer = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 80])
            self.latest_frame = buffer.tobytes()

    def start_server(self):
        """간단한 웹 서버 시작"""
        import http.server
        import socketserver

        class StreamHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self_inner):
                if self_inner.path == "/stream.jpg":
                    if self_inner.server.streamer.latest_frame is not None:
                        self_inner.send_response(200)
                        self_inner.send_header("Content-type", "image/jpeg")
                        self_inner.send_header("Cache-Control", "no-cache")
                        self_inner.end_headers()
                        with self_inner.server.streamer.frame_lock:
                            self_inner.wfile.write(
                                self_inner.server.streamer.latest_frame
                            )
                    else:
                        self_inner.send_response(404)
                        self_inner.end_headers()
                else:
                    self_inner.send_response(200)
                    self_inner.send_header("Content-type", "text/html")
                    self_inner.end_headers()
                    html = """
                    <html lang=ko>
                    <head><title>Red Line Following Car</title></head>
                    <body>
                    <h1>빨간색 선 추적 자동차</h1>
                    <p>빨간색 테이프나 선을 바닥에 붙여주세요!</p>
                    <img src="/stream.jpg" style="max-width:100%;" />
                    <script>
                    setInterval(function() {
                        document.querySelector('img').src = '/stream.jpg?' + new Date().getTime();
                    }, 100);
                    </script>
                    </body>
                    </html>
                    """
                    self_inner.wfile.write(html.encode())

            def log_message(self_inner, format, *args):
                pass  # 로그 출력 비활성화

        try:
            with socketserver.TCPServer(("", self.port), StreamHandler) as httpd:
                httpd.streamer = self
                logger.info(f"웹 스트리밍 서버 시작: http://localhost:{self.port}")
                httpd.serve_forever()
        except Exception as e:
            logger.error(f"웹 서버 시작 실패: {e}")


# ----------------------------------------------------------------------
# RedLineFollowingCar 본체
# ----------------------------------------------------------------------
class RedLineFollowingCar:
    def __init__(
        self,
        camera_index=0,
        headless=True,
        enable_streaming=True,
        skip_camera=False,  # ★ 테스트/Windows용
    ):
        self.headless = headless
        self.enable_streaming = enable_streaming
        self.running = False

        # 모터 컨트롤러 초기화
        try:
            self.motor = MotorController()
        except Exception as e:
            logger.error(f"모터 컨트롤러 초기화 실패: {e}")
            self.motor = None

        # 카메라 초기화 (skip_camera=True면 건너뜀)
        self.cap = None
        if not skip_camera:
            self.cap = cv.VideoCapture(camera_index)
            if not self.cap.isOpened():
                logger.error("카메라를 열 수 없습니다")
                raise Exception("Camera not available")
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv.CAP_PROP_FPS, 30)

        # 웹 스트리밍 설정 (카메라가 있는 실기에서만 사용)
        if self.enable_streaming and not skip_camera:
            self.streamer = WebStreamer()
            self.stream_thread = None
        else:
            self.streamer = None
            self.stream_thread = None

        # 상태 변수
        self.line_center_x = 320  # 기본 중앙값
        self.line_history = deque(maxlen=5)  # 최근 5프레임 기록
        self.lost_count = 0  # 선을 놓친 프레임 수

        # 빨간색 HSV 범위 (조명에 따라 조정 가능)
        self.red_lower1 = np.array([0, 50, 50])  # 빨간색 범위1 (낮은 H)
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 50, 50])  # 빨간색 범위2 (높은 H)
        self.red_upper2 = np.array([180, 255, 255])

        self.stats = {
            "frame_count": 0,
            "start_time": time.time(),
            "last_log_time": time.time(),
            "red_detected_count": 0,
        }

        # 시그널 핸들러 (테스트 환경에서 등록 실패 시 무시)
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except Exception:
            pass

        logger.info("빨간색 선 추적 자동차 초기화 완료 (skip_camera=%s)", skip_camera)
        if skip_camera:
            logger.info("테스트 모드: 카메라 없이 동작")

    # ------------------------------------------------------------------
    def _signal_handler(self, signum, frame):
        logger.info("종료 신호 수신")
        self.running = False

    # ------------------------------------------------------------------
    def detect_red_line(self, img):
        """빨간색 선 검출"""
        height, width = img.shape[:2]

        # ROI: 화면 아래쪽 절반
        roi_start_y = int(height * 0.5)
        roi = img[roi_start_y:height, :]

        # HSV 변환
        hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

        # 빨간색 마스크 (두 범위 합치기)
        mask1 = cv.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv.bitwise_or(mask1, mask2)

        # 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)
        red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)

        # 블러
        red_mask = cv.GaussianBlur(red_mask, (5, 5), 0)

        # 테스트에서도 시각화 필요하므로 항상 복사
        vis_img = img.copy()

        # 무게중심
        moments = cv.moments(red_mask)
        line_detected = False
        red_center_x = width // 2

        if moments["m00"] > 500:  # 충분한 빨간 픽셀
            red_center_x = int(moments["m10"] / moments["m00"])
            line_detected = True
            self.stats["red_detected_count"] += 1

            # 시각화
            red_contours, _ = cv.findContours(
                red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            cv.drawContours(
                vis_img, red_contours, -1, (0, 255, 0), 2, offset=(0, roi_start_y)
            )
            center_y = roi_start_y + int(red_mask.shape[0] * 0.8)
            cv.circle(vis_img, (red_center_x, center_y), 10, (0, 255, 255), -1)
            cv.putText(
                vis_img,
                "RED CENTER",
                (red_center_x - 50, center_y - 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )
        else:
            cv.putText(
                vis_img,
                "RED LINE NOT FOUND!",
                (50, 100),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        return red_center_x, line_detected, vis_img, red_mask

    # ------------------------------------------------------------------
    def calculate_line_center(self, red_center_x, line_detected):
        """선 중심 계산 (히스토리 사용)"""
        if line_detected:
            self.line_history.append(red_center_x)
            self.lost_count = 0
        else:
            self.lost_count += 1
            if self.line_history:
                red_center_x = np.mean(list(self.line_history))

        # 평활화
        if len(self.line_history) >= 3:
            smoothed_center = np.mean(list(self.line_history)[-3:])
        else:
            smoothed_center = red_center_x

        return smoothed_center

    # ------------------------------------------------------------------
    def control_car(self, line_center_x, img_width):
        """자동차 제어"""
        center_x = img_width // 2
        error = line_center_x - center_x

        tolerance = 25

        direction_prefix = "" if self.motor else "SIMULATION: "

        # 선을 너무 오래 놓쳤으면 정지
        if self.lost_count > 30:
            if self.motor:
                self.motor.stop()
            direction = f"{direction_prefix}정지 (빨간 선 없음)"

        # 직진
        elif abs(error) < tolerance:
            if self.motor:
                self.motor.move_forward(speed=55)
            direction = f"{direction_prefix}직진"

        # 좌회전
        elif error < -tolerance:
            if abs(error) > 80:
                if self.motor:
                    self.motor.sharp_left(speed=45)
                direction = f"{direction_prefix}급좌회전"
            else:
                if self.motor:
                    self.motor.turn_left(speed=50)
                direction = f"{direction_prefix}좌회전"

        # 우회전
        else:
            if abs(error) > 80:
                if self.motor:
                    self.motor.sharp_right(speed=45)
                direction = f"{direction_prefix}급우회전"
            else:
                if self.motor:
                    self.motor.turn_right(speed=50)
                direction = f"{direction_prefix}우회전"

        return direction, error

    # ------------------------------------------------------------------
    def add_visualization(self, img, direction, error, line_center_x):
        """시각화 정보 추가"""
        if img is None:
            return None

        height, width = img.shape[:2]

        status_color = (0, 255, 0) if self.lost_count < 5 else (0, 0, 255)
        texts = [
            f"Direction: {direction}",
            f"Error: {error:.1f}px",
            f"Line Center: {line_center_x:.1f}",
            f"Lost Count: {self.lost_count}",
            f"Frame: {self.stats['frame_count']}",
            f"Red Detected: {self.stats['red_detected_count']}",
        ]
        for i, text in enumerate(texts):
            y_pos = 30 + i * 25
            cv.putText(
                img, text, (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2
            )

        # 화면 중앙 (파란선)
        cv.line(img, (width // 2, 0), (width // 2, height), (255, 0, 0), 2)
        # 검출 중심 (노란선)
        cv.line(
            img, (int(line_center_x), 0), (int(line_center_x), height), (0, 255, 255), 3
        )

        # 허용 범위 표시
        tolerance = 25
        left_bound = width // 2 - tolerance
        right_bound = width // 2 + tolerance
        cv.line(
            img, (left_bound, height - 50), (right_bound, height - 50), (0, 255, 0), 5
        )
        cv.putText(
            img,
            "OK ZONE",
            (left_bound, height - 60),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        return img

    # ------------------------------------------------------------------
    def log_status(self, direction, error, line_center_x):
        """상태 로깅 (실행 루프 내에서 주기적 호출)"""
        current_time = time.time()
        if (
            current_time - self.stats["last_log_time"] > 2.0
            and self.stats["frame_count"] > 0
        ):
            fps = self.stats["frame_count"] / (current_time - self.stats["start_time"])
            detection_rate = (
                self.stats["red_detected_count"] / self.stats["frame_count"] * 100
            )
            logger.info(
                f"상태 - 방향: {direction}, 오차: {error:.1f}, "
                f"빨간선중앙: {line_center_x:.1f}, FPS: {fps:.1f}, "
                f"검출률: {detection_rate:.1f}%, 놓친횟수: {self.lost_count}"
            )
            self.stats["last_log_time"] = current_time

    # ------------------------------------------------------------------
    def run(self):
        """메인 실행 루프 (실제 카메라 있는 경우)"""
        if self.cap is None:
            raise RuntimeError("카메라가 초기화되지 않았습니다 (skip_camera=True).")

        logger.info("빨간색 선 추적 자동차 시작")
        logger.info("바닥에 빨간색 테이프나 선을 붙여주세요!")
        logger.info("Ctrl+C로 종료하거나 'stop_car.py' 실행")

        # 웹 스트리밍 서버
        if self.enable_streaming and self.streamer:
            self.stream_thread = threading.Thread(
                target=self.streamer.start_server, daemon=True
            )
            self.stream_thread.start()
            logger.info("웹 스트리밍 시작됨")

        self.running = True
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("카메라에서 프레임을 읽을 수 없습니다")
                    break

                red_center_x, line_detected, vis_img, mask = self.detect_red_line(frame)
                line_center_x = self.calculate_line_center(red_center_x, line_detected)
                direction, error = self.control_car(line_center_x, frame.shape[1])

                if vis_img is not None:
                    vis_img = self.add_visualization(
                        vis_img, direction, error, line_center_x
                    )
                    if self.enable_streaming and self.streamer:
                        self.streamer.update_frame(vis_img)

                self.log_status(direction, error, line_center_x)
                self.stats["frame_count"] += 1

                time.sleep(0.05)  # 20 FPS

        except Exception as e:
            logger.error(f"실행 중 오류: {e}")
        finally:
            self.cleanup()

    # ------------------------------------------------------------------
    def cleanup(self):
        """리소스 정리"""
        logger.info("시스템 종료 중...")
        self.running = False

        if self.motor:
            self.motor.stop()
            time.sleep(0.1)
            self.motor.cleanup()

        if self.cap:
            self.cap.release()

        logger.info("시스템 종료 완료")


# ----------------------------------------------------------------------
# 독립 실행 / 테스트 함수
# ----------------------------------------------------------------------
def create_stop_script():
    """실행 중인 자동차 프로세스를 종료하는 보조 스크립트 생성"""
    stop_script = """#!/usr/bin/env python3
import os
import signal
import psutil

def stop_car():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                cmdline = ' '.join(proc.info['cmdline'])
                if 'red_line_following' in cmdline or 'lane_following' in cmdline:
                    print(f"자동차 프로세스 종료: {proc.info['pid']}")
                    os.kill(proc.info['pid'], signal.SIGTERM)
                    return True
        except:
            continue
    print("자동차 프로세스를 찾을 수 없습니다")
    return False

if __name__ == "__main__":
    stop_car()
"""
    with open("stop_car.py", "w") as f:
        f.write(stop_script)
    os.chmod("stop_car.py", 0o755)
    logger.info("stop_car.py 스크립트 생성됨")


def test_red_detection(image_path):
    """이미지 파일 하나로 빨간선 검출 테스트 (카메라 불필요)"""
    img = cv.imread(image_path)
    if img is None:
        logger.error("이미지를 불러올 수 없습니다")
        return

    car = RedLineFollowingCar(headless=True, enable_streaming=False, skip_camera=True)
    red_center_x, line_detected, _, mask = car.detect_red_line(img)

    logger.info(f"테스트 결과 - 빨간선 검출: {line_detected}, 중심 X: {red_center_x}")

    if line_detected:
        direction, error = car.control_car(red_center_x, img.shape[1])
        logger.info(f"제어 결과 - 방향: {direction}, 오차: {error:.1f}")


if __name__ == "__main__":
    try:
        import psutil  # noqa: F401
    except ImportError:
        logger.warning("psutil 패키지가 없습니다. pip install psutil 실행")

    create_stop_script()

    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            image_path = (
                sys.argv[2] if len(sys.argv) > 2 else str(BASE_DIR / "test_image.jpg")
            )
            test_red_detection(image_path)
        elif sys.argv[1] == "no-stream":
            car = RedLineFollowingCar(
                camera_index=0, headless=True, enable_streaming=False
            )
            car.run()
    else:
        car = RedLineFollowingCar(camera_index=0, headless=True, enable_streaming=True)
        car.run()
