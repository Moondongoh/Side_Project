import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
import pydicom

BASE_DATA_PATH = r"D:\gachon\manifest-1654812109500\Duke-Breast-Cancer-MRI"
ANNOTATION_FILE_PATH = r"D:\gachon\Annotation_Boxes.xlsx"
OUTPUT_DIR = r"D:\gachon\cropped_roi_images3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INCLUSIVE_COORDS = True
USE_IMAGE_POSITION = True

"""DICOM 파일들을 메타를 기준으로 정렬한 뒤 3D 볼륨으로 스택."""


def read_series_sorted(dicom_files):
    metas = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            if USE_IMAGE_POSITION and hasattr(ds, "ImagePositionPatient"):
                z = float(ds.ImagePositionPatient[2])
            else:
                z = float(getattr(ds, "InstanceNumber", 0))
            metas.append((z, f))
        except Exception:
            continue

    if not metas:
        return None, None

    metas.sort(key=lambda x: x[0])

    arrays = []
    for _, f in metas:
        ds = pydicom.dcmread(f)
        arr = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + inter

        if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
            arr = arr.max() - arr

        arrays.append(arr)

    vol = np.stack(arrays, axis=0)
    return vol, metas


def clip_range(start, end, limit, inclusive):
    """슬라이싱 범위를 배열 경계에 안전하게 맞춤."""
    if inclusive:
        end = (
            end + 1
        )  # ***************>>>>> 끝 포함 -> 파이썬 슬라이싱 끝 배타이므로 +1
    start = max(0, int(start))
    end = min(int(end), limit)
    if end < start:
        end = start
    return start, end


try:
    annotations_df = pd.read_excel(ANNOTATION_FILE_PATH)
except FileNotFoundError:
    print(f"오류: {ANNOTATION_FILE_PATH} 파일을 찾을 수 없습니다.")
    sys.exit(1)
except Exception as e:
    print(f"Excel 파일 읽기 오류: {e}")
    sys.exit(1)

total_count = len(annotations_df)
processed_count = 0
skipped_count = 0

for _, row in annotations_df.iterrows():
    patient_id = row["Patient ID"]
    start_r, end_r = row["Start Row"], row["End Row"]
    start_c, end_c = row["Start Column"], row["End Column"]
    start_s, end_s = row["Start Slice"], row["End Slice"]

    patient_processed_successfully = False

    patient_folder = os.path.join(BASE_DATA_PATH, patient_id)
    if os.path.isdir(patient_folder):
        study_folders = glob.glob(os.path.join(patient_folder, "*/"))
        if study_folders:
            study_path = study_folders[0]

            series_folders = glob.glob(os.path.join(study_path, "*-ax*pass-*/"))
            if not series_folders:
                series_folders = glob.glob(os.path.join(study_path, "*-ax*-*/"))

            if series_folders:
                series_path = series_folders[0]
                dicom_files = glob.glob(os.path.join(series_path, "*.dcm"))

                if dicom_files:
                    volume_3d, metas = read_series_sorted(dicom_files)

                    if volume_3d is not None:
                        S, H, W = volume_3d.shape
                        s0, s1 = clip_range(start_s - 1, end_s - 1, S, inclusive=True)
                        r0, r1 = clip_range(
                            start_r, end_r, H, inclusive=INCLUSIVE_COORDS
                        )
                        c0, c1 = clip_range(
                            start_c, end_c, W, inclusive=INCLUSIVE_COORDS
                        )

                        cropped = volume_3d[s0:s1, r0:r1, c0:c1]

                        if cropped.size > 0:
                            patient_output_dir = os.path.join(OUTPUT_DIR, patient_id)
                            os.makedirs(patient_output_dir, exist_ok=True)

                            num_saved = 0
                            for i in range(cropped.shape[0]):
                                img = cropped[i].astype(np.float32)
                                img = cv2.normalize(
                                    img, None, 0, 255, cv2.NORM_MINMAX
                                ).astype(np.uint8)

                                original_slice_number = (s0 + i) + 1
                                fn = f"{patient_id}_slice_{original_slice_number:03d}.png"
                                cv2.imwrite(os.path.join(patient_output_dir, fn), img)
                                num_saved += 1

                            if num_saved > 0:
                                patient_processed_successfully = True

    if patient_processed_successfully:
        processed_count += 1
    else:
        skipped_count += 1


print("\n--- 모든 작업 완료 ---")
print(f"총인원수: {total_count}")
print(f"건너뛴 환자 수 (1,2...): {skipped_count}")
print(f"폴더에 분류된 환자의 수: {processed_count}")
