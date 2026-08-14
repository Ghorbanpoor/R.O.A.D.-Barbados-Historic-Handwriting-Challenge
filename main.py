import os
import re
import cv2
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Config
# =========================================================
class CFG:
    train_csv = "Train.csv"
    test_csv = "Test.csv"
    sample_sub_csv = "SampleSubmission.csv"   # optional if exists
    image_dir = "images"                      # extracted images.zip here

    img_height = 64
    img_width = 384

    batch_size = 32
    num_workers = 2
    epochs = 25
    lr = 1e-3
    seed = 42

    valid_size = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    id_col = "ID"
    text_col = None          # auto-detect from Train.csv
    image_col = None         # auto-detect if available, else use ID-based filename

    checkpoint_path = "best_crnn_ctc.pth"
    submission_path = "submission.csv"


# =========================================================
# Seed
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG.seed)


# =========================================================
# Text utils
# =========================================================
def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def edit_distance(seq1, seq2):
    n = len(seq1)
    m = len(seq2)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)

    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,      # deletion
                dp[i, j - 1] + 1,      # insertion
                dp[i - 1, j - 1] + cost  # substitution
            )
    return dp[n, m]


def word_error_rate(ref, hyp):
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def char_error_rate(ref, hyp):
    ref_chars = list(normalize_text(ref))
    hyp_chars = list(normalize_text(hyp))
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def weighted_score(refs, hyps):
    """
    Competition statement:
    - weighted WER
    - weighted CER
    - final = 0.5 * weighted_WER + 0.5 * weighted_CER

    We weight by reference length so longer references matter more.
    For WER we use number of reference words.
    For CER we use number of reference characters.
    """
    total_word_weight = 0.0
    total_char_weight = 0.0
    sum_weighted_wer = 0.0
    sum_weighted_cer = 0.0

    for ref, hyp in zip(refs, hyps):
        ref = normalize_text(ref)
        hyp = normalize_text(hyp)

        word_w = max(1, len(ref.split()))
        char_w = max(1, len(ref))

        wer = word_error_rate(ref, hyp)
        cer = char_error_rate(ref, hyp)

        sum_weighted_wer += word_w * wer
        sum_weighted_cer += char_w * cer
        total_word_weight += word_w
        total_char_weight += char_w

    weighted_wer = sum_weighted_wer / max(1.0, total_word_weight)
    weighted_cer = sum_weighted_cer / max(1.0, total_char_weight)
    final_score = 0.5 * weighted_wer + 0.5 * weighted_cer
    return final_score, weighted_wer, weighted_cer


# =========================================================
# Auto-detect columns
# =========================================================
def detect_text_column(df, id_col="ID"):
    candidates = [c for c in df.columns if c != id_col]
    if len(candidates) == 1:
        return candidates[0]

    priority = [
        "transcription", "label", "text", "target", "word", "words",
        "sentence", "ground_truth", "gt"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for p in priority:
        if p in lower_map:
            return lower_map[p]

    for c in df.columns:
        if c != id_col and df[c].dtype == object:
            return c

    raise ValueError("Could not detect target text column in Train.csv")


def detect_image_column(df):
    priority = ["image", "img", "image_path", "path", "filename", "file", "img_path"]
    lower_map = {c.lower(): c for c in df.columns}
    for p in priority:
        if p in lower_map:
            return lower_map[p]
    return None


def resolve_image_path(row, image_dir, id_col="ID", image_col=None):
    if image_col is not None and image_col in row and pd.notna(row[image_col]):
        candidate = os.path.join(image_dir, str(row[image_col]))
        if os.path.exists(candidate):
            return candidate

    image_id = str(row[id_col])

    candidates = [
        os.path.join(image_dir, image_id),
        os.path.join(image_dir, image_id + ".jpg"),
        os.path.join(image_dir, image_id + ".jpeg"),
        os.path.join(image_dir, image_id + ".png"),
        os.path.join(image_dir, image_id + ".JPG"),
        os.path.join(image_dir, image_id + ".JPEG"),
        os.path.join(image_dir, image_id + ".PNG"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(f"Image file not found for ID={image_id}")


# =========================================================
# Vocabulary
# =========================================================
class Charset:
    def __init__(self, texts):
        chars = set()
        for t in texts:
            chars.update(list(normalize_text(t)))

        self.blank_token = "<BLANK>"
        self.idx2char = [self.blank_token] + sorted(chars)
        self.char2idx = {ch: i for i, ch in enumerate(self.idx2char)}

    def encode(self, text):
        text = normalize_text(text)
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode_ctc(self, indices):
        result = []
        prev = None
        for idx in indices:
            if idx != 0 and idx != prev:
                result.append(self.idx2char[idx])
            prev = idx
        return "".join(result)

    @property
    def num_classes(self):
        return len(self.idx2char)


# =========================================================
# Image preprocessing
# =========================================================
def resize_with_padding(img, target_h, target_w):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.ones((target_h, target_w), dtype=np.uint8) * 255
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


def preprocess_image(img_path, is_train=True):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # Historical handwriting usually benefits from keeping aspect ratio
    img = resize_with_padding(img, CFG.img_height, CFG.img_width)

    if is_train:
        if random.random() < 0.3:
            alpha = 1.0 + random.uniform(-0.15, 0.15)
            beta = random.uniform(-12, 12)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        if random.random() < 0.2:
            noise = np.random.normal(0, 6, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img, dtype=torch.float32)


# =========================================================
# Dataset
# =========================================================
class HandwritingDataset(Dataset):
    def __init__(self, df, charset, is_train=True):
        self.df = df.reset_index(drop=True)
        self.charset = charset
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = resolve_image_path(
            row=row,
            image_dir=CFG.image_dir,
            id_col=CFG.id_col,
            image_col=CFG.image_col
        )
        image = preprocess_image(img_path, is_train=self.is_train)

        sample_id = str(row[CFG.id_col])

        if self.is_train:
            text = normalize_text(row[CFG.text_col])
            target = torch.tensor(self.charset.encode(text), dtype=torch.long)
            return image, target, text, sample_id

        return image, sample_id


def train_collate_fn(batch):
    images, targets, texts, sample_ids = zip(*batch)
    images = torch.stack(images, dim=0)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    targets_concat = torch.cat(targets) if len(targets) > 0 else torch.tensor([], dtype=torch.long)
    return images, targets_concat, target_lengths, list(texts), list(sample_ids)


def test_collate_fn(batch):
    images, sample_ids = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(sample_ids)


# =========================================================
# Model: CRNN + CTC
# =========================================================
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x384 -> 32x192

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x192 -> 16x96

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 16x96 -> 8x96

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 8x96 -> 4x96
        )

        self.rnn = nn.LSTM(
            input_size=512 * 4,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1
        )

        self.classifier = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)              # [B, 512, 4, W]
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)    # [B, W, 512, 4]
        x = x.contiguous().view(b, w, c * h)  # [B, W, 2048]
        x, _ = self.rnn(x)           # [B, W, 512]
        x = self.classifier(x)       # [B, W, C]
        return x


# =========================================================
# Decode
# =========================================================
@torch.no_grad()
def greedy_decode(logits, charset):
    pred_indices = logits.argmax(dim=-1).cpu().numpy()
    predictions = []
    for seq in pred_indices:
        predictions.append(charset.decode_ctc(seq))
    return predictions


# =========================================================
# Train / Validate / Predict
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for images, targets, target_lengths, texts, sample_ids in loader:
        images = images.to(CFG.device)
        targets = targets.to(CFG.device)
        target_lengths = target_lengths.to(CFG.device)

        optimizer.zero_grad()

        logits = model(images)
        log_probs = F.log_softmax(logits, dim=-1)

        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=log_probs.size(1),
            dtype=torch.long,
            device=CFG.device
        )

        loss = criterion(
            log_probs.permute(1, 0, 2),
            targets,
            input_lengths,
            target_lengths
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate(model, loader, criterion, charset):
    model.eval()
    total_loss = 0.0
    all_refs = []
    all_hyps = []

    for images, targets, target_lengths, texts, sample_ids in loader:
        images = images.to(CFG.device)
        targets = targets.to(CFG.device)
        target_lengths = target_lengths.to(CFG.device)

        logits = model(images)
        log_probs = F.log_softmax(logits, dim=-1)

        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=log_probs.size(1),
            dtype=torch.long,
            device=CFG.device
        )

        loss = criterion(
            log_probs.permute(1, 0, 2),
            targets,
            input_lengths,
            target_lengths
        )
        total_loss += loss.item()

        preds = greedy_decode(logits, charset)
        all_refs.extend(texts)
        all_hyps.extend(preds)

    score, weighted_wer, weighted_cer = weighted_score(all_refs, all_hyps)
    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, score, weighted_wer, weighted_cer, all_refs, all_hyps


@torch.no_grad()
def predict_test(model, loader, charset):
    model.eval()
    ids = []
    preds = []

    for images, sample_ids in loader:
        images = images.to(CFG.device)
        logits = model(images)
        batch_preds = greedy_decode(logits, charset)

        ids.extend(sample_ids)
        preds.extend(batch_preds)

    return ids, preds


# =========================================================
# Main
# =========================================================
def main():
    print(f"Using device: {CFG.device}")

    train_df = pd.read_csv(CFG.train_csv)
    test_df = pd.read_csv(CFG.test_csv)

    if CFG.id_col not in train_df.columns:
        raise ValueError(f"{CFG.id_col} not found in Train.csv")
    if CFG.id_col not in test_df.columns:
        raise ValueError(f"{CFG.id_col} not found in Test.csv")

    if CFG.text_col is None:
        CFG.text_col = detect_text_column(train_df, id_col=CFG.id_col)

    if CFG.image_col is None:
        CFG.image_col = detect_image_column(train_df)

    print(f"Detected text column: {CFG.text_col}")
    print(f"Detected image column: {CFG.image_col}")

    train_df[CFG.text_col] = train_df[CFG.text_col].map(normalize_text)
    train_df = train_df[train_df[CFG.text_col].str.len() > 0].reset_index(drop=True)

    tr_df, va_df = train_test_split(
        train_df,
        test_size=CFG.valid_size,
        random_state=CFG.seed,
        shuffle=True
    )

    charset = Charset(tr_df[CFG.text_col].tolist())
    print(f"Vocabulary size (including blank): {charset.num_classes}")
    print(f"Train samples: {len(tr_df)} | Valid samples: {len(va_df)} | Test samples: {len(test_df)}")

    train_dataset = HandwritingDataset(tr_df, charset, is_train=True)
    valid_dataset = HandwritingDataset(va_df, charset, is_train=True)
    test_dataset = HandwritingDataset(test_df, charset, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        collate_fn=train_collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        collate_fn=train_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        collate_fn=test_collate_fn
    )

    model = CRNN(num_classes=charset.num_classes).to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    best_score = float("inf")

    for epoch in range(1, CFG.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_score, val_wer, val_cer, refs, hyps = validate(model, valid_loader, criterion, charset)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"score={val_score:.4f} | "
            f"wWER={val_wer:.4f} | "
            f"wCER={val_cer:.4f}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "charset": charset.idx2char,
                    "text_col": CFG.text_col,
                    "image_col": CFG.image_col
                },
                CFG.checkpoint_path
            )
            print(f"Saved best model to {CFG.checkpoint_path}")

    print(f"Best validation score: {best_score:.4f}")

    checkpoint = torch.load(CFG.checkpoint_path, map_location=CFG.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    pred_ids, pred_texts = predict_test(model, test_loader, charset)

    # Avoid empty predictions because competition says missing/empty will be penalized
    pred_texts = [p if isinstance(p, str) and len(p.strip()) > 0 else " " for p in pred_texts]

    submission = pd.DataFrame({
        "ID": pred_ids,
        "transcription": pred_texts
    })

    # If sample submission exists, align column names/order to it
    if os.path.exists(CFG.sample_sub_csv):
        sample_sub = pd.read_csv(CFG.sample_sub_csv)
        sub_cols = sample_sub.columns.tolist()

        if len(sub_cols) >= 2:
            pred_col = sub_cols[1]
            submission = submission.rename(columns={"transcription": pred_col})
            submission = sample_sub[[CFG.id_col]].merge(submission, on=CFG.id_col, how="left")
            submission[pred_col] = submission[pred_col].fillna(" ")
        else:
            submission = submission.rename(columns={"transcription": "Target"})

    submission.to_csv(CFG.submission_path, index=False)
    print(f"Submission saved to: {CFG.submission_path}")


if __name__ == "__main__":
    main()
