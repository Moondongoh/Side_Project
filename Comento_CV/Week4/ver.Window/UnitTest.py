import unittest
import numpy as np
import cv2 as cv
import sys
import types

# ------------------------------------------------------------------
# RPi.GPIO Mock를 *먼저* 등록해야 red_line import 시 사용됨
# ------------------------------------------------------------------
# 패키지 stub
mock_rpi = types.ModuleType("RPi")
mock_gpio = types.ModuleType("GPIO")
mock_gpio.BCM = mock_gpio.OUT = None
mock_gpio.setmode = lambda x: None
mock_gpio.setup = lambda pins, mode: None
mock_gpio.PWM = lambda pin, freq: type(
    "PWM",
    (),
    {
        "start": lambda self, x: None,
        "ChangeDutyCycle": lambda self, x: None,
        "stop": lambda self: None,
    },
)()
mock_gpio.cleanup = lambda: None

# sys.modules 등록
sys.modules["RPi"] = mock_rpi
sys.modules["RPi.GPIO"] = mock_gpio
mock_rpi.GPIO = mock_gpio  # 서브모듈 연결

# 이제 안전하게 import
from red_line import RedLineFollowingCar  # noqa: E402


class MockMotorController:
    def __init__(self):
        self.commands = []

    def stop(self):
        self.commands.append("stop")

    def cleanup(self):
        self.commands.append("cleanup")

    def move_forward(self, speed=60):
        self.commands.append("forward")

    def turn_left(self, speed=50):
        self.commands.append("left")

    def turn_right(self, speed=50):
        self.commands.append("right")

    def sharp_left(self, speed=50):
        self.commands.append("sharp_left")

    def sharp_right(self, speed=50):
        self.commands.append("sharp_right")


class TestRedLineFollowingCar(unittest.TestCase):
    def setUp(self):
        self.car = RedLineFollowingCar(
            headless=True, enable_streaming=False, skip_camera=True
        )
        self.car.motor = MockMotorController()

    def test_detect_red_line_with_red_line(self):
        # 빨간색 직선 이미지 생성
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv.line(img, (320, 240), (320, 479), (0, 0, 255), 10)  # BGR 빨간색 선

        center_x, detected, vis_img, mask = self.car.detect_red_line(img)
        self.assertTrue(detected)
        self.assertIsInstance(center_x, int)
        self.assertIsNotNone(vis_img)
        self.assertIsNotNone(mask)

    def test_detect_red_line_without_red_line(self):
        # 빨간색 없는 이미지
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        center_x, detected, vis_img, mask = self.car.detect_red_line(img)
        self.assertFalse(detected)
        self.assertIsInstance(center_x, int)
        self.assertIsNotNone(vis_img)
        self.assertIsNotNone(mask)

    def test_calculate_line_center_with_history(self):
        centers = [300, 310, 320]
        for c in centers:
            self.car.calculate_line_center(c, True)
        smoothed = self.car.calculate_line_center(330, True)
        self.assertTrue(isinstance(smoothed, float))

    def test_calculate_line_center_without_detection(self):
        self.car.line_history.extend([300, 310, 320])
        smoothed = self.car.calculate_line_center(0, False)
        self.assertAlmostEqual(smoothed, np.mean([300, 310, 320]), places=1)

    def test_control_car_behavior(self):
        width = 640
        center = width // 2

        # 정지 조건
        self.car.lost_count = 31
        direction, error = self.car.control_car(center, width)
        self.assertIn("정지", direction)
        self.assertEqual(error, 0)

        # 직진 조건
        self.car.lost_count = 0
        direction, error = self.car.control_car(center + 10, width)
        self.assertIn("직진", direction)

        # 좌회전 조건 (작은 편차)
        direction, error = self.car.control_car(center - 30, width)
        self.assertIn("좌회전", direction)

        # 우회전 조건 (작은 편차)
        direction, error = self.car.control_car(center + 30, width)
        self.assertIn("우회전", direction)

        # 급좌회전 조건 (큰 편차)
        direction, error = self.car.control_car(center - 100, width)
        self.assertIn("급좌회전", direction)

        # 급우회전 조건 (큰 편차)
        direction, error = self.car.control_car(center + 100, width)
        self.assertIn("급우회전", direction)

    def test_add_visualization_returns_image(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        direction = "직진"
        error = 0
        line_center_x = 320
        result = self.car.add_visualization(img, direction, error, line_center_x)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, img.shape)

    def test_cleanup_calls_motor_methods(self):
        mock_motor = MockMotorController()
        self.car.motor = mock_motor
        self.car.cleanup()
        self.assertIn("stop", mock_motor.commands)
        self.assertIn("cleanup", mock_motor.commands)


if __name__ == "__main__":
    unittest.main()
