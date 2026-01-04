import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.seq_transfomer import SketchTransformer

# =========================
# Config
# =========================
MAX_LEN = 100
BATCH_SIZE = 64
NUM_EPOCHS = 50
NUM_CLASSES = 345
LR = 5e-4

TRAIN_SKETCH_LIST = "../dataset/QuickDraw414k/picture_files/tiny_train_set.txt"
TEST_SKETCH_LIST  = "../dataset/QuickDraw414k/picture_files/tiny_test_set.txt"

TRAIN_ROOT = "../dataset/QuickDraw414k/coordinate_files/train"
TEST_ROOT  = "../dataset/QuickDraw414k/coordinate_files/test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Sketch preprocessing
# =========================
def process_one_sketch(npy_path):
    raw = np.load(npy_path, allow_pickle=True, encoding="latin1")

    if isinstance(raw, np.ndarray) and raw.ndim == 2:
        data = raw
    else:
        data = raw.item() if raw.shape == () else raw[0]

    coord = data[:, :2].astype(np.float32)
    pen_down = data[:, 2].astype(np.float32)

    valid = (data[:, 2] + data[:, 3]) > 0
    stroke_len = int(valid.sum())

    return coord, pen_down, stroke_len

# =========================
# Masks
# =========================
def generate_attention_mask(stroke_length):
    mask = torch.zeros((MAX_LEN, MAX_LEN), dtype=torch.float32)
    mask[stroke_length:, :] = -1e8
    mask[:, stroke_length:] = -1e8
    return mask

def generate_padding_mask(stroke_length):
    mask = torch.ones((MAX_LEN, 1), dtype=torch.float32)
    mask[stroke_length:, :] = 0
    return mask

# =========================
# Dataset
# =========================
class QuickdrawDataset(Dataset):
    def __init__(self, sketch_list_path, root_dir):
        self.root_dir = root_dir
        self.samples = []

        with open(sketch_list_path, "r") as f:
            for line in f:
                png_path, label = line.strip().split()
                npy_rel_path = png_path.replace("png", "npy")
                self.samples.append((npy_rel_path, int(label)))

        print(f"Loaded {len(self.samples)} samples from {sketch_list_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_rel_path, label = self.samples[idx]
        npy_path = os.path.join(self.root_dir, npy_rel_path)

        coord, flag_bits, stroke_len = process_one_sketch(npy_path)

        # 截断/填充到 MAX_LEN
        if coord.shape[0] > MAX_LEN:
            coord = coord[:MAX_LEN]
            flag_bits = flag_bits[:MAX_LEN]
            stroke_len = MAX_LEN
        else:
            pad_len = MAX_LEN - coord.shape[0]
            coord = np.pad(coord, ((0, pad_len), (0, 0)), mode="constant")
            flag_bits = np.pad(flag_bits, (0, pad_len), mode="constant")

        coord = torch.from_numpy(coord).float() / 255.0  # 归一化到 [0,1]
        flag_bits = torch.from_numpy(flag_bits).float().unsqueeze(1)

        # masks & position encoding
        attention_mask = generate_attention_mask(stroke_len)
        padding_mask = generate_padding_mask(stroke_len)
        position_encoding = torch.arange(MAX_LEN, dtype=torch.float32).unsqueeze(1)

        return coord, label, flag_bits, stroke_len, attention_mask, padding_mask, position_encoding

# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for coordinate, label, flag_bits, stroke_len, attention_mask, padding_mask, position_encoding in dataloader:
        coordinate = coordinate.to(DEVICE)
        flag_bits = flag_bits.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        padding_mask = padding_mask.to(DEVICE)
        label = label.to(DEVICE)

        logits = model(coordinate, flag_bits, padding_mask, attention_mask)
        loss = criterion(logits, label)

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)

    return total_loss / len(dataloader), correct / total

# =========================
# Main
# =========================
def main():
    # =========================
    # Dataset & DataLoader
    # =========================
    train_dataset = QuickdrawDataset(TRAIN_SKETCH_LIST, TRAIN_ROOT)
    test_dataset  = QuickdrawDataset(TEST_SKETCH_LIST, TEST_ROOT)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # =========================
    # Model
    # =========================
    model = SketchTransformer(embed_dim=128, num_heads=4, num_layers=4, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    #训练中断后这里调参可以加载之前保存的模型恢复训练
    best_test_acc = 0.0
    model.load_state_dict(torch.load("best_model.pth"))
    ###
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for coordinate, label, flag_bits, stroke_len, attention_mask, padding_mask, position_encoding in pbar:
            coordinate = coordinate.to(DEVICE)
            flag_bits = flag_bits.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            padding_mask = padding_mask.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            logits = model(coordinate, flag_bits, padding_mask, attention_mask)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

            pbar.set_postfix(loss=total_loss/(total+1e-6), acc=correct/total)

        scheduler.step()

        test_loss, test_acc = evaluate(model, test_loader, criterion)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch}: Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

    print("Training finished.")

if __name__ == "__main__":
    main()
