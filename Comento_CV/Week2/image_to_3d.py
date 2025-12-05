import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_2d_to_3d(image_path, scale_z=1.0):
    """
    흑백 이미지의 밝기를 기반으로 (x, y, z)로 변환
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    height, width = img.shape
    x = np.arange(0, width)
    y = np.arange(0, height)
    x, y = np.meshgrid(x, y)
    z = img.astype(np.float32) * scale_z

    return x, y, z


def visualize_3d(x, y, z):
    """
    3D 표면 시각화
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, cmap="viridis", edgecolor="none")
    ax.set_title("2D Image to 3D Surface")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Brightness → Z")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 현재 파일의 디렉토리 기준으로 sample.png 경로 구성
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치
    image_path = os.path.join(
        BASE_DIR, "input_image", "images.png"
    )  # 상위 디렉토리의 sample.png

    x, y, z = convert_2d_to_3d(image_path, scale_z=0.5)
    visualize_3d(x, y, z)
