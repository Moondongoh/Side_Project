import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix

# ============================================================
# 1. 경로 및 공통 설정
# ============================================================

# Dataset 루트 (folder_structure.txt 기준)
DATASET_ROOT = r"D:\Foot\Dataset"

# 결과 저장 루트
RESULT_ROOT = r"D:\Foot\Results_kfold"

# 사용할 version / 타입 조합 정의
# folder_structure.txt 구조 기준으로 맞춤
DATASET_CONFIGS = {
    "v1-5000": [
        "_lp(x)-rp(y)-5000",
        "_n-n+1_total-5000",
        "_rp(n)-rp(n+1)-5000",
    ],
    "v2-2048": [
        "_lp(x)-rp(y)_2048",
        "_n-n+1_total_2048",
        "_rp(n)-rp(n+1)_2048",
    ],
    "v3-1024": [
        "_lp(x)-rp(y)_1024",
        "_n-n+1_total_1024",
        "_rp(n)-rp(n+1)_1024",
    ],
}

N_SPLITS = 5
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ============================================================
# 2. Dataset 정의 (CSV 대신 DataFrame 직접 사용)
# ============================================================


class FootpressDataset(Dataset):
    def __init__(self, df, transform=None):
        # df: columns = [path, subject, label]
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            img = self.transform(img)
        return img, label


# 이미지 변환 (ResNet / CNN 공통)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# ============================================================
# 3. 모델 정의
# ============================================================


def build_resnet50():
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # 224x224x3
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_simplecnn():
    return SimpleCNN(num_classes=2).to(device)


# ============================================================
# 4. 공통 train / eval / metric 함수
# ============================================================


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def eval_with_confusion(model, loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # confusion_matrix: [[TN, FP], [FN, TP]] for labels {0,1}
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()

    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # recall for class 1 (Pt)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # recall for class 0 (Co)

    return acc, sensitivity, specificity, (TN, FP, FN, TP)


# ============================================================
# 5. 한 데이터셋(vX + 타입1개)에 대해 K-Fold 수행
# ============================================================


def run_kfold_for_dataset(version, ds_name):
    """
    version: 'v1-5000' / 'v2-2048' / 'v3-1024'
    ds_name: '_lp(x)-rp(y)-5000' 같은 폴더명
    """
    dataset_dir = os.path.join(DATASET_ROOT, version, ds_name)
    print(f"\n\n=== [DATASET] {version} / {ds_name} ===")
    print(f"CSV dir: {dataset_dir}")

    train_csv = os.path.join(dataset_dir, "train.csv")
    val_csv = os.path.join(dataset_dir, "val.csv")
    test_csv = os.path.join(dataset_dir, "test.csv")

    if not (
        os.path.exists(train_csv)
        and os.path.exists(val_csv)
        and os.path.exists(test_csv)
    ):
        print(f"[WARN] CSV files not found in {dataset_dir}, skip.")
        return

    # train+val 합쳐서 TrainVal로 사용 (subject-wise KFold 용)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"TrainVal: {len(trainval_df)} images, Test: {len(test_df)} images")
    print(f"Unique subjects in TrainVal: {trainval_df['subject'].nunique()}")
    print(f"Unique subjects in Test    : {test_df['subject'].nunique()}")

    # StratifiedGroupKFold 설정 (subject 기준 group, label 기준 stratify)
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    X = trainval_df["path"].values  # dummy, 실제로는 안 씀
    y = trainval_df["label"].values
    groups = trainval_df["subject"].values

    # 결과 저장 폴더
    base_save_dir = os.path.join(RESULT_ROOT, version, ds_name)
    os.makedirs(base_save_dir, exist_ok=True)

    # fold별 + 모델별 로그를 모으는 list
    rows = []

    fold_iter = sgkf.split(X, y, groups)

    for fold, (tr_idx, val_idx) in enumerate(fold_iter):
        print(f"\n--- Fold {fold} ---")

        fold_train_df = trainval_df.iloc[tr_idx].reset_index(drop=True)
        fold_val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

        print(
            f"Fold {fold} Train size: {len(fold_train_df)}, Val size: {len(fold_val_df)}"
        )
        print(
            f"  Train subjects: {fold_train_df['subject'].nunique()}, "
            f"Val subjects: {fold_val_df['subject'].nunique()}"
        )

        # 공통 Dataloader
        train_loader = DataLoader(
            FootpressDataset(fold_train_df, transform),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            FootpressDataset(fold_val_df, transform),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        test_loader = DataLoader(
            FootpressDataset(test_df, transform),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        # 두 모델(ResNet50, SimpleCNN)을 순서대로 돌린다.
        for model_name in ["resnet50", "simplecnn"]:
            print(f"\n[MODEL] {model_name} | Fold {fold}")

            if model_name == "resnet50":
                model = build_resnet50()
            else:
                model = build_simplecnn()

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LR)

            best_val_acc = 0.0
            best_epoch = -1

            # 모델 저장 경로
            model_save_dir = os.path.join(base_save_dir, model_name)
            os.makedirs(model_save_dir, exist_ok=True)
            best_model_path = os.path.join(model_save_dir, f"best_model_fold{fold}.pt")

            # 에폭 루프
            for epoch in range(1, EPOCHS + 1):
                train_loss, train_acc = train_one_epoch(
                    model, train_loader, optimizer, criterion
                )
                val_loss, val_acc = evaluate(model, val_loader, criterion)

                print(
                    f"[{model_name}] Fold {fold} | Epoch [{epoch}/{EPOCHS}] "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} "
                    f"| Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
                )

                # 베스트 모델 갱신
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_model_path)

                # 로그 한 줄 저장
                rows.append(
                    {
                        "version": version,
                        "dataset": ds_name,
                        "model": model_name,
                        "fold": fold,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    }
                )

            # 에폭 다 돈 뒤, best 모델 로드해서 테스트셋 평가
            print(
                f"[{model_name}] Fold {fold} best epoch: {best_epoch}, best val acc: {best_val_acc:.4f}"
            )
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            test_acc, test_sens, test_spec, (TN, FP, FN, TP) = eval_with_confusion(
                model, test_loader
            )

            print(
                f"[{model_name}] Fold {fold} Test Acc: {test_acc:.4f}, "
                f"Sens: {test_sens:.4f}, Spec: {test_spec:.4f}"
            )
            print(
                f"Confusion Matrix [[TN, FP], [FN, TP]] = [[{TN}, {FP}], [{FN}, {TP}]]"
            )

            # test 결과도 추가로 rows에 요약 한 줄 남겨두기 (epoch -1로 표시)
            rows.append(
                {
                    "version": version,
                    "dataset": ds_name,
                    "model": model_name,
                    "fold": fold,
                    "epoch": -1,  # test 요약 표시
                    "train_loss": None,
                    "train_acc": None,
                    "val_loss": None,
                    "val_acc": best_val_acc,
                    "test_acc": test_acc,
                    "test_sensitivity": test_sens,
                    "test_specificity": test_spec,
                    "TN": TN,
                    "FP": FP,
                    "FN": FN,
                    "TP": TP,
                }
            )

    # 모든 fold 끝나면, 로그를 CSV로 저장
    log_df = pd.DataFrame(rows)

    log_path = os.path.join(base_save_dir, "kfold_train_log.csv")
    log_df.to_csv(log_path, index=False, encoding="utf-8-sig")
    print(f"\n[LOG SAVED] {log_path}")


# ============================================================
# 6. 메인: 모든 v1/v2/v3 × 3타입 자동 수행
# ============================================================


def main():
    for version, ds_list in DATASET_CONFIGS.items():
        for ds_name in ds_list:
            run_kfold_for_dataset(version, ds_name)


if __name__ == "__main__":
    main()
