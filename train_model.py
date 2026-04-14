import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# --- config ---
CSV_PATH = "listings_clean.csv"
IMAGES_DIR = "images"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
MODEL_SAVE_PATH = "price_model.pth"

# use gpu if available, otherwise cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {DEVICE}")


# --- dataset ---

class HousePriceDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

        # log-transform prices so the model learns relative differences better
        # (a $100k difference near $200k is more meaningful than near $2M)
        self.prices = np.log1p(self.df["price"].values).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = int(self.df.loc[idx, "img_id"])
        img_path = os.path.join(self.images_dir, f"{img_id}.jpg")

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        price = self.prices[idx]
        return img, price


# --- transforms ---

# training transform includes random flips/crops to help the model generalize
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# validation/test transform is simpler — no random augmentation
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# --- model ---

def build_model():
    # use a pretrained resnet50 as the backbone
    # it already knows how to detect edges, textures, shapes from imagenet training
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # freeze the early layers so we don't overwrite the pretrained knowledge
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False

    # replace the final classification head with a regression head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1)
    )

    return model


# --- training loop ---

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for imgs, prices in loader:
        imgs = imgs.to(DEVICE)
        prices = prices.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, prices)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(imgs)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for imgs, prices in loader:
            imgs = imgs.to(DEVICE)
            prices = prices.to(DEVICE).unsqueeze(1)

            preds = model(imgs)
            loss = criterion(preds, prices)
            total_loss += loss.item() * len(imgs)

    return total_loss / len(loader.dataset)


# --- main ---

def main():
    # load csv and drop rows missing an image id or price
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["img_id", "price"])
    df["img_id"] = df["img_id"].astype(int)

    # only keep rows where the image file actually exists
    df = df[df["img_id"].apply(
        lambda x: os.path.exists(os.path.join(IMAGES_DIR, f"{x}.jpg"))
    )]
    print(f"dataset size after filtering: {len(df)} rows")

    # split into train and validation sets (80/20)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = HousePriceDataset(train_df, IMAGES_DIR, transform=train_transform)
    val_dataset   = HousePriceDataset(val_df,   IMAGES_DIR, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model().to(DEVICE)

    # mean absolute error on log prices is a smooth loss for regression
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    # reduce learning rate if validation loss stops improving
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss   = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        # convert log-scale mae back to a dollar-scale approximation for readability
        approx_dollar_mae = np.expm1(val_loss)

        print(f"epoch {epoch:02d}/{EPOCHS} | "
              f"train loss: {train_loss:.4f} | "
              f"val loss: {val_loss:.4f} | "
              f"approx val MAE: ${approx_dollar_mae:,.0f}")

        # save the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> saved new best model to {MODEL_SAVE_PATH}")

    print("\ntraining complete.")
    print(f"best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()