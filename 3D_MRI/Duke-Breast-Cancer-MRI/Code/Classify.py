import pandas as pd
import os
import shutil

excel_path = r"경로지정"
src_root = r"경로지정"
dst_root = r"경로지정"

df = pd.read_excel(excel_path)

id_col = "Patient Information"
label_col = "Tumor Response"

selected = df[df[label_col].isin([1, 2])]

patient_ids = selected[id_col].astype(str).tolist()
print("추출 대상 환자:", patient_ids)

os.makedirs(dst_root, exist_ok=True)

count = 0
for pid in patient_ids:
    folder_name = f"{pid.zfill(3)}"
    src_path = os.path.join(src_root, folder_name)
    dst_path = os.path.join(dst_root, folder_name)

    if os.path.exists(src_path):
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        print(f"[OK] {folder_name} 복사 완료")
        count += 1
    else:
        print(f"[MISS] {folder_name} 없음")

print(f"\n총 {count}개 환자 폴더 추출 완료!")
