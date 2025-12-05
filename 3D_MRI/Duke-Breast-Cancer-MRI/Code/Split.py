import os
import shutil
import random

IMG_ROOT = r"D:\gachon\selected_patients"
OUT_ROOT = r"D:\gachon\split_6_2_2"
SEED = 42

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2


def main():
    random.seed(SEED)

    all_patients = [
        d for d in os.listdir(IMG_ROOT) if os.path.isdir(os.path.join(IMG_ROOT, d))
    ]
    all_patients = sorted(all_patients)

    n = len(all_patients)
    print(f"총 환자 폴더 수: {n}")
    if n == 0:
        raise RuntimeError("IMG_ROOT에 폴더가 없습니다. 경로를 확인하세요.")

    random.shuffle(all_patients)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_patients = all_patients[:n_train]
    val_patients = all_patients[n_train : n_train + n_val]
    test_patients = all_patients[n_train + n_val :]

    print(f"Train 환자 수: {len(train_patients)}")
    print(f"Val   환자 수: {len(val_patients)}")
    print(f"Test  환자 수: {len(test_patients)}")

    assert len(set(train_patients) & set(val_patients)) == 0
    assert len(set(train_patients) & set(test_patients)) == 0
    assert len(set(val_patients) & set(test_patients)) == 0

    train_root = os.path.join(OUT_ROOT, "train")
    val_root = os.path.join(OUT_ROOT, "val")
    test_root = os.path.join(OUT_ROOT, "test")

    os.makedirs(train_root, exist_ok=True)
    os.makedirs(val_root, exist_ok=True)
    os.makedirs(test_root, exist_ok=True)

    def copy_group(patients, dst_root, group_name):
        for i, pid in enumerate(patients, start=1):
            src = os.path.join(IMG_ROOT, pid)
            dst = os.path.join(dst_root, pid)

            shutil.copytree(src, dst, dirs_exist_ok=True)
            if i % 10 == 0 or i == len(patients):
                print(f"[{group_name}] {i}/{len(patients)} 복사 완료")

    print("\n=== Train 복사 ===")
    copy_group(train_patients, train_root, "train")

    print("\n=== Val 복사 ===")
    copy_group(val_patients, val_root, "val")

    print("\n=== Test 복사 ===")
    copy_group(test_patients, test_root, "test")

    print("\n완료!")
    print(f"Train 폴더: {train_root}")
    print(f"Val   폴더: {val_root}")
    print(f"Test  폴더: {test_root}")


if __name__ == "__main__":
    main()
