import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
import cv2
import numpy as np
from tqdm import tqdm

# --- AYARLAR ---
CIHAZ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESIM_BOYUTU = 640  # Roboflow'daki boyutunla aynı olmalı
BATCH_SIZE = 4  # Bilgisayarın kasarsa 2 yapabilirsin
EPOCHS = 100  # Test için düşük tutabilirsin
LR = 0.0001  # Öğrenme hızı


# --- DATASET SINIFI ---
class OtonomDataset(Dataset):
    def __init__(self, resim_listesi, resim_klasoru, maske_klasoru):
        self.resimler = resim_listesi
        self.resim_klasoru = resim_klasoru
        self.maske_klasoru = maske_klasoru

    def __len__(self):
        return len(self.resimler)

    def __getitem__(self, idx):
        resim_adi = self.resimler[idx]
        resim_yolu = os.path.join(self.resim_klasoru, resim_adi)
        maske_yolu = os.path.join(self.maske_klasoru, resim_adi.replace(".jpg", ".png"))

        # Resim yükleme ve normalize etme
        resim = cv2.imread(resim_yolu)
        resim = cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)
        resim = cv2.resize(resim, (RESIM_BOYUTU, RESIM_BOYUTU))
        resim = resim.transpose(2, 0, 1).astype('float32') / 255.0

        # Maske yükleme
        maske = cv2.imread(maske_yolu, cv2.IMREAD_GRAYSCALE)
        maske = cv2.resize(maske, (RESIM_BOYUTU, RESIM_BOYUTU))
        maske = np.where(maske > 127, 1, 0).astype('float32')
        maske = np.expand_dims(maske, axis=0)

        return torch.from_numpy(resim), torch.from_numpy(maske)


# --- EĞİTİM FONKSİYONU ---
def egitim_yap(model, loader, optimizer, loss_fn):
    model.train()
    toplam_loss = 0
    for resimler, maskeler in tqdm(loader, desc="Eğitiliyor"):
        resimler, maskeler = resimler.to(CIHAZ), maskeler.to(CIHAZ)
        optimizer.zero_grad()
        cikti = model(resimler)
        loss = loss_fn(cikti, maskeler)
        loss.backward()
        optimizer.step()
        toplam_loss += loss.item()
    return toplam_loss / len(loader)


# --- ANA DÖNGÜ (5-FOLD) ---
resim_klasoru = "islenmisVeriler/tumResimler"
maske_klasoru = "islenmisVeriler/tumMaskeler"
tum_resimler = np.array(os.listdir(resim_klasoru))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(tum_resimler)):
    print(f"\n{'=' * 20} FOLD {fold + 1} BAŞLIYOR {'=' * 20}")

    train_resimler = tum_resimler[train_idx]
    val_resimler = tum_resimler[val_idx]

    train_ds = OtonomDataset(train_resimler, resim_klasoru, maske_klasoru)
    val_ds = OtonomDataset(val_resimler, resim_klasoru, maske_klasoru)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Model: U-Net + ResNet18 (Hızlı ve etkili)
    model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=1).to(CIHAZ)

    # Loss: Dice Loss (Segmentasyon için en iyisi)
    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        ortalama_loss = egitim_yap(model, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {ortalama_loss:.4f}")

    # Modeli Kaydet
    model_adi = f"ciktiDosyalari/modeller/unet_fold{fold + 1}.pth"
    torch.save(model.state_dict(), model_adi)
    print(f"Model kaydedildi: {model_adi}")