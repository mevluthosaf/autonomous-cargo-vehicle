import sys
import os

# Proje Kök Dizini ve Sanal Ortam (.venv) Kontrolü
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

# Eğer betik .venv dışındaki bir Python yorumlayıcısıyla çağrıldıysa, otomatik olarak .venv ile yeniden başlat:
if os.path.exists(VENV_PYTHON) and os.path.realpath(sys.executable) != os.path.realpath(VENV_PYTHON):
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

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

# --- PID HIZ VE YÖN KONTROLCÜSÜ ---
class PIDController:
    """
    Otonom şerit ve yol takibi için PID kontrolcüsü.
    error = yol_merkezi_x - goruntu_merkezi_x
    """
    def __init__(self, kp=0.45, ki=0.01, kd=0.15, base_speed=150, min_speed=50, max_speed=220, windup_limit=50):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.base_speed = base_speed
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.windup_limit = windup_limit

        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None

    def compute(self, error):
        now = time.time()
        if self.last_time is None:
            dt = 0.5
        else:
            dt = max(now - self.last_time, 0.2)
        self.last_time = now

        # Oransal bileşen (P)
        p = self.kp * error

        # İntegral bileşen (I) - Anti-windup korumalı
        self.integral += error * dt
        self.integral = max(-self.windup_limit, min(self.windup_limit, self.integral))
        i = self.ki * self.integral

        # Türev bileşen (D)
        derivative = (error - self.prev_error) / dt
        d = self.kd * derivative
        self.prev_error = error

        # Toplam PID yön düzeltmesi
        correction = p + i + d

        # Diferansiyel Hız Hesabı:
        # error > 0: Yol sağda -> Araç sağa dönmeli -> Sol motor hızlanır, Sağ motor yavaşlar
        # error < 0: Yol solda -> Araç sola dönmeli -> Sağ motor hızlanır, Sol motor yavaşlar
        left_speed = int(self.base_speed + correction)
        right_speed = int(self.base_speed - correction)

        # Hız sınırlandırma (clamping: min_speed .. max_speed)
        left_speed = max(self.min_speed, min(self.max_speed, left_speed))
        right_speed = max(self.min_speed, min(self.max_speed, right_speed))

        return left_speed, right_speed, correction

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None


pid = PIDController(kp=0.45, ki=0.01, kd=0.15, base_speed=150)


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
print(">>> OTONOM SÜRÜŞ BAŞLADI (PID MODU) <<<")
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
                left_speed, right_speed, correction = pid.compute(error)
                cmd_packet = f"<{left_speed},{right_speed}>\n"

                # Durum metni belirleme (görsel telemetri)
                if abs(error) <= 25:
                    yon = "DUZ      "
                elif error > 0:
                    yon = "HAFIF SAG" if error < 80 else "SERT SAG "
                else:
                    yon = "HAFIF SOL" if error > -80 else "SERT SOL "

                telemetri = (
                    f"[# {islem_sayaci:04d}] Yön: {yon} | Hata: {error:>4}px | "
                    f"Düzeltme: {correction:>+5.1f} | PWM (Sol/Sağ): {left_speed:3d}/{right_speed:3d}"
                )
            else:
                pid.reset()
                cmd_packet = "<0,0>\n"
                telemetri = f"[# {islem_sayaci:04d}] Yön: YOL YOK   | Hata:  YOK | PWM (Sol/Sağ):   0/  0 (FREN)"

            # Arduino'ya paket gönderme (<left,right>)
            if arduino:
                arduino.write(cmd_packet.encode())
                print(telemetri)
            else:
                print(f"{telemetri} | (ARDUINO YOK)")

except KeyboardInterrupt:
    print("\n" + "=" * 40)
    print("[-] Kullanıcı komutuyla (Ctrl+C) sistem durduruluyor...")
finally:
    cap.release()
    if arduino:
        # Program kapanırken motorları durdur
        arduino.write(b"<0,0>\n")
        arduino.close()
    print("[*] Donanım bağlantıları güvenle kapatıldı. İyi çalışmalar!")
    print("=" * 40)