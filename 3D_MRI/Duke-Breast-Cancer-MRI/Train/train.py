# import os
# import glob
# import random
# import warnings

# import cv2
# import numpy as np
# import pandas as pd

# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
# from sklearn.utils.class_weight import compute_class_weight

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import models, transforms
# import argparse

# warnings.filterwarnings("ignore")

# # =========================
# # 라벨 매핑(Tumor Response): 1->0(neg), 2->1(pos)
# # 1번 반응, 2번 미반응
# # 0이 반응 1이 미반응
# # =========================
# B_TO_LABEL = {1: 0, 2: 1}


# # =========================
# # 유틸리티
# # =========================
# def set_seed(seed: int = 42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def build_manifest(root_dir: str) -> pd.DataFrame:
#     rows = []
#     for pid_dir in sorted(glob.glob(os.path.join(root_dir, "*"))):
#         if not os.path.isdir(pid_dir):
#             continue
#         pid = os.path.basename(pid_dir)
#         pngs = sorted(glob.glob(os.path.join(pid_dir, "*.png")))
#         for p in pngs:
#             rows.append({"patient_id": pid, "img_path": p})
#     return pd.DataFrame(rows)


# class SliceDataset(Dataset):
#     def __init__(self, df_rows: pd.DataFrame, aug: bool = False, img_size: int = 224):
#         self.df = df_rows.reset_index(drop=True)
#         self.aug = aug
#         self.img_size = img_size

#         base = [
#             transforms.ToTensor(),
#             transforms.Resize((self.img_size, self.img_size)),
#         ]
#         aug_tf = [
#             transforms.ToTensor(),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomRotation(degrees=10),
#             transforms.Resize((self.img_size, self.img_size)),
#         ]
#         self.tf_base = transforms.Compose(base)
#         self.tf_aug = transforms.Compose(aug_tf)

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         path = row["img_path"]
#         label = int(row["label"])

#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             raise RuntimeError(f"이미지 읽기 실패: {path}")

#         if self.aug:
#             x = self.tf_aug(img)
#         else:
#             x = self.tf_base(img)

#         x = x.repeat(3, 1, 1)
#         return x, label, row["patient_id"], path


# def make_loader(df_rows, bs=64, aug=False, shuffle=False,
#                 num_workers=0, pin_memory=False, img_size=224):
#     ds = SliceDataset(df_rows, aug=aug, img_size=img_size)
#     return DataLoader(
#         ds,
#         batch_size=bs,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         persistent_workers=(num_workers > 0),
#     )


# def metrics_from_probs(probs, ys, th=0.5):
#     preds = (probs >= th).astype(int)
#     acc = accuracy_score(ys, preds)
#     f1  = f1_score(ys, preds)
#     try:
#         auc = roc_auc_score(ys, probs)
#     except ValueError:
#         auc = float("nan")
#     return acc, f1, auc, preds


# def run_epoch(model, loader, criterion, optimizer, device, train=True):
#     model.train(train)
#     losses, probs_all, ys_all, pids_all = [], [], [], []

#     for x, y, pids, _ in loader:
#         x = x.to(device, non_blocking=True)
#         y = y.to(device, non_blocking=True)

#         with torch.set_grad_enabled(train):
#             logits = model(x)
#             loss = criterion(logits, y)
#             if train:
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#         losses.append(loss.item())
#         prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
#         probs_all.extend(prob)
#         ys_all.extend(y.cpu().numpy().tolist())
#         pids_all.extend(list(pids))

#     return float(np.mean(losses)), np.array(probs_all), np.array(ys_all), np.array(pids_all)


# # =========================
# # 모델 정의
# # =========================
# class SmallCNN(nn.Module):
#     def __init__(self, num_classes=2):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),

#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Linear(128 * 28 * 28, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256, 64),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(64, num_classes),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x


# # =========================
# # 라벨: Features.xlsx에서 읽기
# # =========================
# def build_label_map_from_excel(label_xlsx):
#     df_lab = pd.read_excel(label_xlsx)

#     id_col   = "Patient Information"
#     resp_col = "Tumor Response"

#     df_lab[id_col] = df_lab[id_col].astype(str).str.strip()
#     df_lab["_resp"] = pd.to_numeric(df_lab[resp_col], errors="coerce")

#     df_lab_b12 = df_lab[df_lab["_resp"].isin([1, 2])][[id_col, "_resp"]].rename(
#         columns={id_col: "patient_id", "_resp": "label_raw"}
#     )
#     df_lab_b12["label"] = df_lab_b12["label_raw"].map(B_TO_LABEL)

#     return df_lab_b12[["patient_id", "label"]]


# def attach_labels(df_manifest, df_label_map):
#     df_manifest["patient_id"] = df_manifest["patient_id"].astype(str).str.strip()
#     df_label_map["patient_id"] = df_label_map["patient_id"].astype(str).str.strip()
#     return df_manifest.merge(df_label_map, on="patient_id", how="inner")


# # =========================
# # 메인
# # =========================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train_root", type=str,
#                         default=r"D:\gachon\split_6_2_2\train")
#     parser.add_argument("--val_root", type=str,
#                         default=r"D:\gachon\split_6_2_2\val")
#     parser.add_argument("--test_root", type=str,
#                         default=r"D:\gachon\split_6_2_2\test")
#     parser.add_argument("--label_xlsx", type=str,
#                         default=r"D:\gachon\Features.xlsx")
#     parser.add_argument("--out_dir", type=str,
#                         default=r"D:\gachon\exp_weighted_training2")
#     parser.add_argument("--model", type=str, default="smallcnn",
#                         choices=["smallcnn", "resnet18"])
#     parser.add_argument("--train_bs", type=int, default=64)
#     parser.add_argument("--val_bs", type=int, default=128)
#     parser.add_argument("--test_bs", type=int, default=128)
#     parser.add_argument("--lr", type=float, default=1e-4)
#     parser.add_argument("--weight_decay", type=float, default=1e-4)
#     parser.add_argument("--num_workers", type=int, default=4)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--img_size", type=int, default=224)
#     args = parser.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)
#     set_seed(args.seed)

#     # 1) 라벨 맵 로드
#     df_label_map = build_label_map_from_excel(args.label_xlsx)

#     # 2) manifest + label join
#     print("Building manifests...")
#     df_tr_manifest = build_manifest(args.train_root)
#     df_va_manifest = build_manifest(args.val_root)
#     df_te_manifest = build_manifest(args.test_root)

#     df_tr = attach_labels(df_tr_manifest, df_label_map)
#     df_va = attach_labels(df_va_manifest, df_label_map)
#     df_te = attach_labels(df_te_manifest, df_label_map)

#     print(f"Patients: train {len(df_tr['patient_id'].unique())}, "
#           f"val {len(df_va['patient_id'].unique())}, "
#           f"test {len(df_te['patient_id'].unique())}")
#     print(f"Images  : train {len(df_tr)}, val {len(df_va)}, test {len(df_te)}")

#     # ============================
#     # 🔥 CLASS DISTRIBUTION 출력
#     # ============================
#     def print_dist(df, name):
#         print(f"\n[{name}]")
#         print(df["label"].value_counts())
#         pos = df["label"].sum()
#         print(f"pos ratio = {pos/len(df):.3f}")

#     print_dist(df_tr, "train")
#     print_dist(df_va, "val")
#     print_dist(df_te, "test")

#     # ============================
#     # 🔥 CLASS WEIGHT 계산
#     # ============================
#     class_weights = compute_class_weight(
#         class_weight="balanced",
#         classes=np.array([0,1]),
#         y=df_tr["label"].values
#     )
#     print("\n[INFO] Computed class weights:", class_weights)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     pin = torch.cuda.is_available()

#     train_loader = make_loader(df_tr, bs=args.train_bs, aug=True,
#                                shuffle=True, num_workers=args.num_workers,
#                                pin_memory=pin, img_size=args.img_size)
#     val_loader   = make_loader(df_va, bs=args.val_bs, aug=False,
#                                shuffle=False, num_workers=args.num_workers,
#                                pin_memory=pin, img_size=args.img_size)
#     test_loader  = make_loader(df_te, bs=args.test_bs, aug=False,
#                                shuffle=False, num_workers=args.num_workers,
#                                pin_memory=pin, img_size=args.img_size)

#     # 모델 선택
#     if args.model == "smallcnn":
#         model = SmallCNN(num_classes=2)
#     else:
#         model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         model.fc = nn.Linear(model.fc.in_features, 2)

#     model.to(device)

#     # 🔥 class weight 적용
#     cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
#     criterion = nn.CrossEntropyLoss(weight=cw_tensor)

#     optimizer = torch.optim.AdamW(model.parameters(),
#                                   lr=args.lr,
#                                   weight_decay=args.weight_decay)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="max", patience=3, factor=0.5
#     )

#     torch.backends.cudnn.benchmark = torch.cuda.is_available()

#     best_f1 = -1.0
#     best_path = os.path.join(args.out_dir, f"best_{args.model}.pt")

#     # ============================
#     # 🔥 TRAINING LOOP
#     # ============================
#     for epoch in range(1, 101):
#         tr_loss, _, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
#         va_loss, va_prob, va_y, va_pid = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

#         va_acc, va_f1, va_auc, va_pred = metrics_from_probs(va_prob, va_y)

#         print(
#             f"[Ep {epoch:03d}] "
#             f"train loss={tr_loss:.4f} | "
#             f"val loss={va_loss:.4f} acc={va_acc:.3f} f1={va_f1:.3f} auc={va_auc:.3f}"
#         )

#         scheduler.step(va_f1)

#         if va_f1 > best_f1:
#             best_f1 = va_f1
#             torch.save(model.state_dict(), best_path)
#             print("  -> Saved best")

#     # ============================
#     # 🔥 TEST EVALUATION
#     # ============================
#     model.load_state_dict(torch.load(best_path, map_location=device))
#     _, te_prob, te_y, te_pid = run_epoch(model, test_loader, criterion, optimizer, device, train=False)

#     te_acc, te_f1, te_auc, te_pred = metrics_from_probs(te_prob, te_y)

#     print("\n[Slice-level]")
#     print(f"ACC={te_acc:.3f} F1={te_f1:.3f} AUC={te_auc:.3f}")
#     print(classification_report(te_y, te_pred, digits=3))
#     print("Confusion matrix:\n", confusion_matrix(te_y, te_pred))

#     # ============================
#     # 🔥 Patient-level 평균 확률 평가
#     # ============================
#     df_te_pred = pd.DataFrame({"patient_id": te_pid, "y": te_y, "prob": te_prob})
#     pt_agg = df_te_pred.groupby("patient_id").agg(
#         y=("y", "first"), prob=("prob", "mean")
#     ).reset_index()

#     pt_pred = (pt_agg["prob"] >= 0.5).astype(int)

#     pt_acc = accuracy_score(pt_agg["y"], pt_pred)
#     pt_f1  = f1_score(pt_agg["y"], pt_pred)
#     try:
#         pt_auc = roc_auc_score(pt_agg["y"], pt_agg["prob"])
#     except ValueError:
#         pt_auc = float("nan")

#     print("\n[Patient-level] (mean prob)")
#     print(f"ACC={pt_acc:.3f} F1={pt_f1:.3f} AUC={pt_auc:.3f}")
#     print("Confusion matrix:\n", confusion_matrix(pt_agg["y"], pt_pred))

#     pt_out = os.path.join(args.out_dir, f"patient_level_{args.model}.csv")
#     pt_agg.assign(pred=pt_pred).to_csv(pt_out, index=False)
#     print(f"\nSaved results to: {pt_out}")


# if __name__ == "__main__":
#     main()


import os
import glob
import random
import warnings

import cv2
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import argparse

warnings.filterwarnings("ignore")

# =========================
# 라벨 매핑(Tumor Response): 1->0(반응), 2->1(미반응)
# =========================
B_TO_LABEL = {1: 0, 2: 1}


# =========================
# 유틸리티
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_manifest(root_dir: str) -> pd.DataFrame:
    rows = []
    for pid_dir in sorted(glob.glob(os.path.join(root_dir, "*"))):
        if not os.path.isdir(pid_dir):
            continue
        pid = os.path.basename(pid_dir)
        pngs = sorted(glob.glob(os.path.join(pid_dir, "*.png")))

        for p in pngs:
            rows.append({"patient_id": pid, "img_path": p})

    return pd.DataFrame(rows)


class SliceDataset(Dataset):
    def __init__(self, df_rows: pd.DataFrame, aug: bool = False, img_size: int = 224):
        self.df = df_rows.reset_index(drop=True)
        self.aug = aug
        self.img_size = img_size

        base = [
            transforms.ToTensor(),
            transforms.Resize((self.img_size, self.img_size)),
        ]
        aug_tf = [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.Resize((self.img_size, self.img_size)),
        ]
        self.tf_base = transforms.Compose(base)
        self.tf_aug = transforms.Compose(aug_tf)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["img_path"]
        label = int(row["label"])

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"이미지 읽기 실패: {path}")

        if self.aug:
            x = self.tf_aug(img)
        else:
            x = self.tf_base(img)

        # 흑백을 3채널로 복제
        x = x.repeat(3, 1, 1)
        return x, label, row["patient_id"], path


def make_loader(df_rows, bs=64, aug=False, shuffle=False,
                num_workers=0, pin_memory=False, img_size=224):

    ds = SliceDataset(df_rows, aug=aug, img_size=img_size)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def metrics_from_probs(probs, ys, th=0.5):
    preds = (probs >= th).astype(int)
    acc = accuracy_score(ys, preds)
    f1 = f1_score(ys, preds)

    try:
        auc = roc_auc_score(ys, probs)
    except ValueError:
        auc = float("nan")

    return acc, f1, auc, preds


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    losses, probs_all, ys_all, pids_all = [], [], [], []

    for x, y, pids, _ in loader:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

        probs_all.extend(prob)
        ys_all.extend(y.cpu().numpy().tolist())
        pids_all.extend(list(pids))

    return float(np.mean(losses)), np.array(probs_all), np.array(ys_all), np.array(pids_all)


# =========================
# 모델 정의
# =========================
class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def build_label_map_from_excel(label_xlsx):
    df_lab = pd.read_excel(label_xlsx)

    id_col = "Patient Information"
    resp_col = "Tumor Response"

    df_lab[id_col] = df_lab[id_col].astype(str).str.strip()
    df_lab["_resp"] = pd.to_numeric(df_lab[resp_col], errors="coerce")

    df_lab_b12 = df_lab[df_lab["_resp"].isin([1, 2])][[id_col, "_resp"]].rename(
        columns={id_col: "patient_id", "_resp": "label_raw"}
    )
    df_lab_b12["label"] = df_lab_b12["label_raw"].map(B_TO_LABEL)

    return df_lab_b12[["patient_id", "label"]]


def attach_labels(df_manifest, df_label_map):
    df_manifest["patient_id"] = df_manifest["patient_id"].astype(str).str.strip()
    df_label_map["patient_id"] = df_label_map["patient_id"].astype(str).str.strip()

    return df_manifest.merge(df_label_map, on="patient_id", how="inner")


# =========================
# 메인
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", default=r"D:\gachon\split_6_2_2_2\train")
    parser.add_argument("--val_root",   default=r"D:\gachon\split_6_2_2_2\val")
    parser.add_argument("--test_root",  default=r"D:\gachon\split_6_2_2_2\test")
    parser.add_argument("--label_xlsx", default=r"D:\gachon\Features.xlsx")
    parser.add_argument("--out_dir",    default=r"D:\gachon\exp_no_weights")
    parser.add_argument("--model",      default="smallcnn", choices=["smallcnn", "resnet18"])
    parser.add_argument("--train_bs",   type=int, default=64)
    parser.add_argument("--val_bs",     type=int, default=128)
    parser.add_argument("--test_bs",    type=int, default=128)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # 라벨 매핑
    df_label_map = build_label_map_from_excel(args.label_xlsx)

    # Manifest 생성
    print("Building manifests...")
    df_tr_manifest = build_manifest(args.train_root)
    df_va_manifest = build_manifest(args.val_root)
    df_te_manifest = build_manifest(args.test_root)

    df_tr = attach_labels(df_tr_manifest, df_label_map)
    df_va = attach_labels(df_va_manifest, df_label_map)
    df_te = attach_labels(df_te_manifest, df_label_map)

    print(f"Patients: train {len(df_tr.patient_id.unique())}, val {len(df_va.patient_id.unique())}, test {len(df_te.patient_id.unique())}")
    print(f"Images  : train {len(df_tr)}, val {len(df_va)}, test {len(df_te)}")

    def print_dist(df, name):
        print(f"\n[{name}]")
        print(df["label"].value_counts())
        pos = df["label"].sum()
        print(f"pos ratio = {pos/len(df):.3f}")

    print_dist(df_tr, "train")
    print_dist(df_va, "val")
    print_dist(df_te, "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = torch.cuda.is_available()

    train_loader = make_loader(df_tr, args.train_bs, aug=True, shuffle=True,
                               num_workers=args.num_workers, pin_memory=pin)
    val_loader = make_loader(df_va, args.val_bs, aug=False,
                             num_workers=args.num_workers, pin_memory=pin)
    test_loader = make_loader(df_te, args.test_bs, aug=False,
                              num_workers=args.num_workers, pin_memory=pin)

    # 모델 선택
    if args.model == "smallcnn":
        model = SmallCNN()
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)

    model.to(device)

    # --------------------------
    # 🔥 클래스 가중치 제거됨
    # --------------------------
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)

    best_f1 = -1
    best_path = os.path.join(args.out_dir, f"best_{args.model}.pt")

    for epoch in range(1, 101):
        tr_loss, _, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_prob, va_y, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        va_acc, va_f1, va_auc, _ = metrics_from_probs(va_prob, va_y)

        print(f"[Ep {epoch:03d}] train={tr_loss:.4f} | val={va_loss:.4f} acc={va_acc:.3f} f1={va_f1:.3f} auc={va_auc:.3f}")

        scheduler.step(va_f1)

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), best_path)
            print("  -> Saved best model")

    # 테스트 평가
    model.load_state_dict(torch.load(best_path, map_location=device))
    _, te_prob, te_y, te_pid = run_epoch(model, test_loader, criterion, optimizer, device, train=False)

    te_acc, te_f1, te_auc, te_pred = metrics_from_probs(te_prob, te_y)

    print("\n[Slice-level]")
    print(f"ACC={te_acc:.3f} F1={te_f1:.3f} AUC={te_auc:.3f}")
    print(classification_report(te_y, te_pred, digits=3))
    print(confusion_matrix(te_y, te_pred))

    # 환자 레벨 평가
    df_te_pred = pd.DataFrame({"patient_id": te_pid, "y": te_y, "prob": te_prob})
    pt_agg = df_te_pred.groupby("patient_id").agg(y=("y", "first"), prob=("prob", "mean")).reset_index()

    pt_pred = (pt_agg["prob"] >= 0.5).astype(int)
    pt_acc = accuracy_score(pt_agg["y"], pt_pred)
    pt_f1 = f1_score(pt_agg["y"], pt_pred)

    try:
        pt_auc = roc_auc_score(pt_agg["y"], pt_agg["prob"])
    except:
        pt_auc = float("nan")

    print("\n[Patient-level]")
    print(f"ACC={pt_acc:.3f} F1={pt_f1:.3f} AUC={pt_auc:.3f}")
    print(confusion_matrix(pt_agg["y"], pt_pred))

    pt_out = os.path.join(args.out_dir, f"patient_level_{args.model}.csv")
    pt_agg.assign(pred=pt_pred).to_csv(pt_out, index=False)
    print(f"\nSaved: {pt_out}")


if __name__ == "__main__":
    main()
