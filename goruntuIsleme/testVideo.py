import torch
import cv2
import numpy as np
import os
import segmentation_models_pytorch as smp
from tqdm import tqdm  # İşlem çubuğu için (pip install tqdm)

# --- AYARLAR ---
CIHAZ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESIM_BOYUTU = 640
THRESHOLD = 0.75  # Taşmayı önlemek için eşiği 0.75 yaptık (Gerekirse 0.8 yapabilirsin)

MODEL_KLASORU = "ciktiDosyalari/modeller/"
GIRIS_VIDEO_YOLU = r"assets/test_video.mp4"  # Test etmek istediğin videonun yolu
CIKTI_KLASORU = "ciktiDosyalari/test_videolari/"

# Çıktı klasörü yoksa otomatik oluştur
os.makedirs(CIKTI_KLASORU, exist_ok=True)

# 1. Modeli Tanımla (İskelet)
model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1).to(CIHAZ)

# 2. 5 Farklı Model İçin Döngü Başlat
for fold in range(1, 6):
    model_adi = f"unet_fold{fold}.pth"
    model_yolu = os.path.join(MODEL_KLASORU, model_adi)

    if not os.path.exists(model_yolu):
        print(f"{model_adi} bulunamadı, atlanıyor...")
        continue

    print(f"\n{'=' * 10} FOLD {fold} VİDEO İŞLEME BAŞLADI {'=' * 10}")

    # Ağırlıkları Yükle
    model.load_state_dict(torch.load(model_yolu, map_location=CIHAZ))
    model.eval()

    # Giriş Videosunu Aç
    cap = cv2.VideoCapture(GIRIS_VIDEO_YOLU)
    if not cap.isOpened():
        print(f"Hata: Video dosyası açılamadı -> {GIRIS_VIDEO_YOLU}")
        continue

    # Orijinal videonun özelliklerini al
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    toplam_kare = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Çıktı Videosunu Ayarla (MP4 formatında kaydeder)
    cikti_video_yolu = os.path.join(CIKTI_KLASORU, f"gorsel_test_video_fold{fold}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cikti_video_yolu, fourcc, fps, (RESIM_BOYUTU, RESIM_BOYUTU))

    # Kare kare işleme döngüsü (tqdm ile ekranda yüklenme barı görünür)
    for _ in tqdm(range(toplam_kare), desc=f"Fold {fold} İşleniyor"):
        ret, frame = cap.read()
        if not ret:
            break  # Video bittiyse döngüden çık

        # 1. Ön İşleme
        frame_resized = cv2.resize(frame, (RESIM_BOYUTU, RESIM_BOYUTU))
        resim_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(resim_rgb.transpose(2, 0, 1)).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(CIHAZ)

        # 2. Model Tahmini (Forward)
        with torch.no_grad():
            cikti = model(input_tensor)
            maske = torch.sigmoid(cikti).squeeze().cpu().numpy()
            maske = (maske > THRESHOLD).astype(np.uint8) * 255

        # 3. Üzerine Yeşil Boya Sürme (Overlay)
        yesil_katman = np.zeros_like(frame_resized)
        yesil_katman[:, :] = [0, 255, 0]  # Yeşil renk katmanı
        boyali_alan = cv2.bitwise_and(yesil_katman, yesil_katman, mask=maske)

        # Orijinal kare ile boyayı karıştır (%40 şeffaflık)
        sonuc_karesi = cv2.addWeighted(frame_resized, 1.0, boyali_alan, 0.4, 0)

        # 4. Kareyi Yeni Videoya Yaz
        out.write(sonuc_karesi)

    # Kaynakları serbest bırak
    cap.release()
    out.release()
    print(f"Fold {fold} videosu başarıyla kaydedildi: {cikti_video_yolu}")

print("\nTüm modeller için video test işlemleri tamamlandı!")