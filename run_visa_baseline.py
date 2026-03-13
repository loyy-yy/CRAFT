import torch
import numpy as np
import os
import cv2
import csv
from pathlib import Path
from PIL import Image, ImageFile
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score  # 🚨 引入硬核计算工具

# 🛡️ 开启底层图片截断容错
ImageFile.LOAD_TRUNCATED_IMAGES = True
from run_patchcore_visa import import_patchcore_and_engine

print("=" * 60)
print("🚀 启动原生 Baseline 实验 (完美 4 栏出图 & 全自动落盘版)")
print("=" * 60)

# 路径配置
DATA_ROOT = Path(r"D:\code\industrial\datasets\VisA_pytorch\1cls")

# ✅ 将成绩单和图片的保存路径统一收束到一个文件夹里
SAVE_DIR = Path(r"D:\code\industrial\results\visa_patchcore_wrn50\ALL_RUN_BASELINE")
FIGURE_DIR = SAVE_DIR
CSV_RESULT_PATH = SAVE_DIR / "Auto_Baseline_Metrics.csv"

if not SAVE_DIR.exists(): SAVE_DIR.mkdir(parents=True)

ALL_CATEGORIES = [d.name for d in DATA_ROOT.iterdir() if d.is_dir()]

data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def safe_load_image(path, is_mask=False):
    try:
        img = Image.open(path)
        return img.convert("L") if is_mask else img.convert("RGB")
    except Exception:
        return Image.new('L' if is_mask else 'RGB', (256, 256), (0))


class MockBatch:
    def __init__(self, images, labels, masks):
        self.image = images
        self.label = labels
        self.gt_label = labels.clone().detach().to(torch.long)
        self.mask = masks
        self.gt_mask = masks.clone().detach().to(torch.long)

    def to(self, device):
        self.image = self.image.to(device)
        self.label = self.label.to(device)
        self.gt_label = self.gt_label.to(device)
        self.mask = self.mask.to(device)
        self.gt_mask = self.gt_mask.to(device)
        return self

    def update(self, **kwargs):
        for key, value in kwargs.items(): setattr(self, key, value)
        return self

    def __getitem__(self, key):
        if isinstance(key, str): return getattr(self, key)
        return {"image": self.image[key], "label": self.label[key]}

    def keys(self):
        return ["image", "label", "gt_label", "mask", "gt_mask"]

    def __len__(self):
        return len(self.image)


def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    masks = torch.stack([item[2] for item in batch])
    return MockBatch(images, labels, masks)


def prepare_data(category):
    train_files = list((DATA_ROOT / category / "train").rglob("*.[jp][pn]g"))
    train_loader = DataLoader(TensorDataset(torch.stack([data_transforms(safe_load_image(p)) for p in train_files]),
                                            torch.zeros(len(train_files), dtype=torch.long),
                                            torch.zeros((len(train_files), 1, 256, 256), dtype=torch.long)),
                              batch_size=4, shuffle=False, collate_fn=custom_collate)

    test_files = list((DATA_ROOT / category / "test").rglob("*.[jp][pn]g"))
    test_tensors, test_labels, test_masks = [], [], []
    gt_root = DATA_ROOT / category / "ground_truth"
    for p in test_files:
        test_tensors.append(data_transforms(safe_load_image(p)))
        is_defect = "good" not in str(p).lower() and "normal" not in str(p).lower()
        test_labels.append(1 if is_defect else 0)
        mask_tensor = torch.zeros((1, 256, 256), dtype=torch.long)
        if is_defect and gt_root.exists():
            stem = p.stem
            potential_masks = [f for f in gt_root.rglob("*") if
                               (stem == f.stem or stem + "_mask" == f.stem) and f.suffix.lower() in ['.png', '.jpg']]
            if potential_masks:
                mask_tensor = (transforms.ToTensor()(
                    transforms.Resize((256, 256))(safe_load_image(potential_masks[0], is_mask=True))) > 0.5).to(
                    torch.long)
        test_masks.append(mask_tensor)
    return train_loader, DataLoader(
        TensorDataset(torch.stack(test_tensors), torch.tensor(test_labels), torch.stack(test_masks)),
        batch_size=4, shuffle=False, collate_fn=custom_collate)


def run_single_category(category, PatchcoreCls, EngineCls):
    print(f"\n⚡ 处理类别: [{category}] ...")
    train_loader, test_loader = prepare_data(category)
    model = PatchcoreCls(backbone="wide_resnet50_2", coreset_sampling_ratio=0.1)
    engine = EngineCls(default_root_dir=str(SAVE_DIR / category), max_epochs=1, accelerator="gpu", devices=1)

    engine.fit(model=model, train_dataloaders=train_loader)
    engine.test(model=model, dataloaders=test_loader)

    predictions = engine.predict(model=model, dataloaders=test_loader)

    # 🚀 手动算出绝对真实的指标，并直接写入硬盘！
    all_amaps = []
    all_masks = []
    for batch in predictions:
        all_amaps.append(batch.anomaly_map.cpu().flatten())
        if hasattr(batch, 'mask') and batch.mask is not None:
            all_masks.append(batch.mask.cpu().flatten())

    best_threshold = 0.5
    best_f1 = 0.0
    pixel_auroc = 0.0

    if len(all_masks) > 0:
        flat_amaps = torch.cat(all_amaps).numpy()
        flat_masks = torch.cat(all_masks).numpy()

        # 1. 算 AUROC
        try:
            pixel_auroc = roc_auc_score(flat_masks, flat_amaps)
        except:
            pass

        # 2. 算最好 F1
        precision, recall, thresholds = precision_recall_curve(flat_masks, flat_amaps)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        # 3. 🚨 立刻将数据安全落盘到 CSV 中！
        file_exists = CSV_RESULT_PATH.exists()
        with open(CSV_RESULT_PATH, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Category (类别)", "Pixel_AUROC (基准)", "Pixel_F1Score (基准真实下限)"])
            writer.writerow([category, f"{pixel_auroc:.4f}", f"{best_f1:.4f}"])

        print(f"✅ [{category}] 成绩已安全存入表格: AUROC={pixel_auroc:.4f}, F1={best_f1:.4f}")

    # ==========================================
    # 🎨 视觉灾难收集 (恢复完美的 4 栏对比图)
    # ==========================================
    saved_bad_count = 0
    saved_good_count = 0
    cmap = plt.get_cmap('jet')

    for batch in predictions:
        for i in range(len(batch.label)):
            is_defect = (batch.label[i] == 1)
            folder_name = "bad" if is_defect else "good"

            # 【第 1 栏：原图】
            img_tensor = batch.image[i].cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_np = (torch.clamp(img_tensor * std + mean, 0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            col1_img = Image.fromarray(img_np).resize((256, 256))

            # 【第 2 栏：真实的黑白标签图 (GT Mask)】
            gt_mask_np = batch.mask[i].cpu().squeeze().numpy()
            col2_img = Image.fromarray((gt_mask_np * 255).astype(np.uint8)).convert("RGB").resize((256, 256))

            # 【第 3 栏：彩色热力图】
            amap_np = batch.anomaly_map[i].cpu().squeeze().numpy()
            amap_norm = (amap_np - amap_np.min()) / (amap_np.max() - amap_np.min() + 1e-8)
            heat_overlay = (img_np * 0.5 + (cmap(amap_norm)[:, :, :3] * 255).astype(np.uint8) * 0.5).astype(np.uint8)
            col3_img = Image.fromarray(heat_overlay).resize((256, 256))

            # 【第 4 栏：带有预测红圈的重叠图】
            mask_overlay = img_np.copy()
            mask_bool = amap_np > best_threshold
            contours, _ = cv2.findContours(((mask_bool) * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_overlay, contours, -1, (255, 0, 0), 2)

            gt_contours, _ = cv2.findContours((gt_mask_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mask_overlay, gt_contours, -1, (0, 255, 0), 1)
            col4_img = Image.fromarray(mask_overlay).resize((256, 256))

            # 🧩 稳稳贴进 1024 宽的画板
            canvas = Image.new('RGB', (1024, 256), (0, 0, 0))
            canvas.paste(col1_img, (0, 0))
            canvas.paste(col2_img, (256, 0))
            canvas.paste(col3_img, (512, 0))
            canvas.paste(col4_img, (768, 0))

            # 图片全部整整齐齐落入指定的 ALL_RUN_BASELINE 文件夹
            save_path = FIGURE_DIR / category / folder_name
            save_path.mkdir(parents=True, exist_ok=True)
            canvas.save(save_path / f"{(saved_bad_count if is_defect else saved_good_count):03d}.jpg")
            if is_defect:
                saved_bad_count += 1
            else:
                saved_good_count += 1


def main():
    Patchcore, Engine = import_patchcore_and_engine()
    # 每次运行前，如果之前有旧表格，先删掉防止数据重复
    if CSV_RESULT_PATH.exists():
        CSV_RESULT_PATH.unlink()

    for cat in ALL_CATEGORIES:
        try:
            run_single_category(cat, Patchcore, Engine)
        except Exception as e:
            print(f"❌ {cat} 报错: {e}")

    print(f"\n🎉 恭喜！所有类别运行完毕。完美成绩单及图片已保存在: {SAVE_DIR}")


if __name__ == "__main__": main()