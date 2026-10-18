import os

# OpenCV'nin gereksiz terminal uyarılarını (GStreamer vb.) tamamen susturur
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

import cv2
import torch
import numpy as np
import serial
import time
import segmentation_models_pytorch as smp

# Donanım ayarı
CIHAZ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 40)
print(f"[*] Sistem Başlatılıyor...")
print(f"[*] Kullanılan Donanım: {str(CIHAZ).upper()}")

# Model yükleme
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "unet_fold1.pth")

try:
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=CIHAZ))
    model.to(CIHAZ)
    model.eval()
    print(f"[+] U-Net Modeli başarıyla yüklendi: {MODEL_PATH}")
except Exception as e:
    print(f"[-] Model yükleme hatası ({MODEL_PATH}): {e}")
    exit()

# Arduino bağlantısı
try:
    arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=0.1)
    time.sleep(2)
    print("[+] Arduino bağlantısı kuruldu (/dev/ttyUSB0).")
except Exception as e:
    print(f"[-] Arduino bağlantı hatası: {e}")
    arduino = None

TOLERANCE = 50


def tahmin_et(kamera_karesi):
    resim = cv2.cvtColor(kamera_karesi, cv2.COLOR_BGR2RGB)
    resim = cv2.resize(resim, (640, 640))
    resim = resim.transpose(2, 0, 1).astype('float32') / 255.0
    tensor = torch.from_numpy(resim).unsqueeze(0).to(CIHAZ)
    with torch.no_grad():
        cikti = model(tensor)
        maske = torch.sigmoid(cikti) > 0.5
    return maske.squeeze().cpu().numpy().astype(np.uint8) * 255


def get_centroid(mask):
    M = cv2.moments(mask)
    if M["m00"] != 0:
        return int(M["m10"] / M["m00"])
    return None


# USB Kamera Başlatma (GStreamer hatalarını önlemek için CAP_V4L2 kullanıldı)
print("[*] Kamera aranıyor...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("[-] HATA: USB Kamera bulunamadı veya yetki yok!")
    exit()
else:
    print("[+] Kamera bağlantısı aktif.")

islem_araligi = 0.5  # 2 FPS için 0.5 saniye bekleme
son_islem_zamani = time.time()
islem_sayaci = 0

print("=" * 40)
print(">>> OTONOM SÜRÜŞ BAŞLADI <<<")
print(">>> İzleme Modu Aktif. Durdurmak için: Ctrl + C <<<")
print("-" * 40)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[-] Kameradan kare okunamadı!")
            break

        su_an = time.time()

        # Performans: Saniyede sadece 2 kareyi derin öğrenme modeline sok
        if su_an - son_islem_zamani >= islem_araligi:
            gecen_sure = su_an - son_islem_zamani
            son_islem_zamani = su_an
            islem_sayaci += 1

            orig_h, orig_w = frame.shape[:2]
            mask_640 = tahmin_et(frame)
            mask_resized = cv2.resize(mask_640, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            road_center_x = get_centroid(mask_resized)
            img_center_x = orig_w // 2

            if road_center_x is not None:
                error = road_center_x - img_center_x
                if abs(error) <= TOLERANCE:
                    cmd = 'S'
                    durum_metni = "DUZ "
                elif error > TOLERANCE:
                    cmd = 'R'
                    durum_metni = "SAG "
                else:
                    cmd = 'L'
                    durum_metni = "SOL "
                hata_miktari = f"{error:>4}"  # Sağa dayalı formatlama
            else:
                cmd = 'B'
                durum_metni = "DUR "
                hata_miktari = "YOK "

            # Komut gönderme
            if arduino:
                arduino.write(cmd.encode())
                print(f"[# {islem_sayaci:04d}] Hedef: {durum_metni} | Hata Payı: {hata_miktari} | Gönderilen: {cmd}")
            else:
                print(f"[# {islem_sayaci:04d}] Hedef: {durum_metni} | Hata Payı: {hata_miktari} | (ARDUINO YOK)")

except KeyboardInterrupt:
    print("\n" + "=" * 40)
    print("[-] Kullanıcı komutuyla (Ctrl+C) sistem durduruluyor...")
finally:
    cap.release()
    if arduino:
        # Program kapanırken aracı güvene almak için durma komutu
        arduino.write('B'.encode())
        arduino.close()
    print("[*] Donanım bağlantıları güvenle kapatıldı. İyi çalışmalar!")
    print("=" * 40)