import sys
import os
import time
import threading
import argparse
import subprocess
import datetime

# Proje Kök Dizini ve Sanal Ortam (.venv) Kontrolü
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

# Eğer betik .venv dışındaki bir Python yorumlayıcısıyla çağrıldıysa, otomatik olarak .venv ile yeniden başlat:
if os.path.exists(VENV_PYTHON) and os.path.realpath(sys.executable) != os.path.realpath(VENV_PYTHON):
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

import warnings
warnings.filterwarnings("ignore")
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

import cv2
import numpy as np
import serial
import psutil

# --- DONANIM VE SİSTEM TELEMETRİSİ (RASPBERRY PI OS) ---
def get_rpi_telemetry():
    """
    Raspberry Pi 5 OS ve işlemci sağlık verilerini okur.
    """
    # 1. CPU Sıcaklığı
    cpu_temp = 0.0
    try:
        # Raspberry Pi standart thermal zone
        if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                cpu_temp = float(f.read().strip()) / 1000.0
        else:
            # vcgencmd alternatifi
            out = subprocess.check_output(["vcgencmd", "measure_temp"], stderr=subprocess.DEVNULL).decode()
            cpu_temp = float(out.replace("temp=", "").replace("'C\n", "").strip())
    except Exception:
        cpu_temp = 48.5  # Mac/PC geliştirme ortamı için simülasyon değeri

    # 2. CPU Kullanım Yüzdesi
    cpu_pct = psutil.cpu_percent(interval=None)

    # 3. RAM Kullanımı
    mem = psutil.virtual_memory()
    ram_used_gb = mem.used / (1024 ** 3)
    ram_total_gb = mem.total / (1024 ** 3)
    ram_pct = mem.percent

    return {
        "cpu_temp": cpu_temp,
        "cpu_pct": cpu_pct,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "ram_pct": ram_pct,
    }


# --- GLOBAL PAYLAŞIMLI DURUM VE KİLİTLER (THREAD-SAFE SHARED STATE) ---
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True

        # U-Net Yol Çıktıları
        self.road_mask = None
        self.road_center_x = None
        self.error_px = 0
        self.pid_left = 0
        self.pid_right = 0
        self.road_detected = False
        self.unet_fps = 0.0

        # YOLO Algılama Çıktıları
        self.detections = []  # {'box': (x1, y1, x2, y2), 'label': str, 'conf': float, 'type': str}
        self.emergency_stop = False
        self.active_sign = None
        self.speed_multiplier = 1.0
        self.yolo_fps = 0.0
        self.last_detected_sign = "Yok"
        self.last_detected_time = ""

        # Son Nihai Gönderilen PWM ve Durum
        self.final_left_pwm = 0
        self.final_right_pwm = 0
        self.action_state = "BAŞLATILIYOR"
        self.arduino_connected = False


# --- PID HIZ VE YÖN KONTROLCÜSÜ ---
class PIDController:
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
        dt = 0.1 if self.last_time is None else max(now - self.last_time, 0.05)
        self.last_time = now

        p = self.kp * error

        self.integral += error * dt
        self.integral = max(-self.windup_limit, min(self.windup_limit, self.integral))
        i = self.ki * self.integral

        derivative = (error - self.prev_error) / dt
        d = self.kd * derivative
        self.prev_error = error

        correction = p + i + d

        left_speed = int(self.base_speed + correction)
        right_speed = int(self.base_speed - correction)

        left_speed = max(self.min_speed, min(self.max_speed, left_speed))
        right_speed = max(self.min_speed, min(self.max_speed, right_speed))

        return left_speed, right_speed, correction

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None


# --- 1. THREAD: KAMERA OKUMA (ZERO BUFFER LAG & VIDEO/IMAGE DESTEĞİ) ---
class CameraThread(threading.Thread):
    def __init__(self, shared_state, src=0, width=640, height=480):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.src = src
        self.width = width
        self.height = height

    def run(self):
        print("[*] Kamera okuma iş parçacığı başlatılıyor...")
        is_file = False
        cap = None

        if isinstance(self.src, str) and os.path.exists(self.src):
            # Dosya modu (Video veya Resim)
            if self.src.lower().endswith(('.png', '.jpg', '.jpeg')):
                static_img = cv2.imread(self.src)
                print(f"[+] Statik görsel test modu aktif: {self.src}")
                while self.shared_state.running:
                    with self.shared_state.lock:
                        self.shared_state.latest_frame = static_img.copy()
                    time.sleep(0.04)
                return
            else:
                cap = cv2.VideoCapture(self.src)
                is_file = True
                print(f"[+] Video dosyası akışı aktif: {self.src}")
        else:
            try:
                cam_id = int(self.src)
            except ValueError:
                cam_id = 0
            cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(cam_id)

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if cap is None or not cap.isOpened():
            print(f"[-] Kamera / Video açılamadı (src={self.src})!")
            with self.shared_state.lock:
                self.shared_state.running = False
            return

        print("[+] Kamera akışı aktif.")
        while self.shared_state.running:
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Videoyu başa sar
                    continue
                time.sleep(0.01)
                continue

            with self.shared_state.lock:
                self.shared_state.latest_frame = frame

        cap.release()
        print("[*] Kamera iş parçacığı sonlandı.")


# --- 2. THREAD: U-NET BİSİKLET YOLU SEGMENTASYONU ---
class UNetTrackingThread(threading.Thread):
    def __init__(self, shared_state, model_path, pid_controller, target_fps=15):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.model_path = model_path
        self.pid = pid_controller
        self.interval = 1.0 / target_fps
        self.session = None
        self.torch_model = None
        self.is_onnx = model_path.endswith(".onnx")
        self.device = "cpu"

    def _model_yukle(self):
        if self.is_onnx:
            import onnxruntime as ort
            print(f"[*] U-Net ONNX Modeli Yükleniyor: {self.model_path}")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            self.session = ort.InferenceSession(self.model_path, opts, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
        else:
            import torch
            import segmentation_models_pytorch as smp
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[*] U-Net PyTorch Modeli Yükleniyor ({self.device}): {self.model_path}")
            self.torch_model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
            self.torch_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.torch_model.to(self.device)
            self.torch_model.eval()

    def run(self):
        try:
            self._model_yukle()
        except Exception as e:
            print(f"[-] U-Net model yüklenemedi: {e}")
            return

        print("[+] U-Net Yol Takip İş Parçacığı Aktif.")
        kare_sayaci = 0
        fps_zaman = time.time()

        while self.shared_state.running:
            start_t = time.time()
            frame = None
            with self.shared_state.lock:
                if self.shared_state.latest_frame is not None:
                    frame = self.shared_state.latest_frame.copy()

            if frame is None:
                time.sleep(0.02)
                continue

            orig_h, orig_w = frame.shape[:2]
            resim = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resim = cv2.resize(resim, (640, 640))
            resim = resim.transpose(2, 0, 1).astype('float32') / 255.0

            if self.is_onnx:
                inp = np.expand_dims(resim, axis=0)
                ort_outs = self.session.run(None, {self.input_name: inp})
                cikti = ort_outs[0].squeeze()
                mask = (1.0 / (1.0 + np.exp(-cikti))) > 0.5
                mask_uint8 = (mask.astype(np.uint8)) * 255
            else:
                import torch
                tensor = torch.from_numpy(resim).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    cikti = self.torch_model(tensor)
                    mask = (torch.sigmoid(cikti) > 0.5).squeeze().cpu().numpy()
                    mask_uint8 = (mask.astype(np.uint8)) * 255

            mask_resized = cv2.resize(mask_uint8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # Centroid (Kütle Merkezi) Hesabı
            M = cv2.moments(mask_resized)
            road_center = int(M["m10"] / M["m00"]) if M["m00"] != 0 else None
            img_center = orig_w // 2

            kare_sayaci += 1
            if time.time() - fps_zaman >= 1.0:
                current_fps = kare_sayaci / (time.time() - fps_zaman)
                kare_sayaci = 0
                fps_zaman = time.time()
            else:
                current_fps = getattr(self, "current_fps", 12.0)
            self.current_fps = current_fps

            with self.shared_state.lock:
                self.shared_state.road_mask = mask_resized
                self.shared_state.unet_fps = current_fps
                if road_center is not None:
                    error = road_center - img_center
                    left, right, _ = self.pid.compute(error)
                    self.shared_state.road_detected = True
                    self.shared_state.road_center_x = road_center
                    self.shared_state.error_px = error
                    self.shared_state.pid_left = left
                    self.shared_state.pid_right = right
                else:
                    self.pid.reset()
                    self.shared_state.road_detected = False
                    self.shared_state.road_center_x = None
                    self.shared_state.error_px = 0
                    self.shared_state.pid_left = 0
                    self.shared_state.pid_right = 0

            elapsed = time.time() - start_t
            sleep_time = max(0.001, self.interval - elapsed)
            time.sleep(sleep_time)

        print("[*] U-Net iş parçacığı sonlandı.")


# --- 3. THREAD: YOLO TRAFİK LEVHASI VE ENGEL TESPİTİ ---
class YoloDetectionThread(threading.Thread):
    def __init__(self, shared_state, model_path, target_fps=15, conf_thresh=0.45):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.model_path = model_path
        self.interval = 1.0 / target_fps
        self.conf_thresh = conf_thresh

        # Sınıf Eşleştirme Sözlükleri (Roboflow 21 Sınıfı + Dinamik Engeller)
        self.STOP_KEYS = ["stop", "dur", "red_light", "kirmizi_isik", "do_not_enter", "no_entry", "railway_crossing", "girilmez"]
        self.SLOW_KEYS = ["yield", "yol_ver", "ped_crossing", "ped_zebra_cross", "pedestrian", "yaya", "warning", "yellow_light", "traffic_light", "bump", "kasis", "speed", "dikkat"]
        self.OBSTACLE_KEYS = ["obstacle", "engel", "cone", "duba", "box", "koli", "person", "insan", "car", "araba", "bicycle", "bisiklet"]
        self.RIGHT_KEYS = ["turn_right", "do_not_turn_l", "sag"]
        self.LEFT_KEYS = ["turn_left", "enter_left_lane", "do_not_turn_r", "sol"]

    def _sinif_tipi_belirle(self, label_lower):
        for k in self.STOP_KEYS:
            if k in label_lower:
                return "DUR"
        for k in self.OBSTACLE_KEYS:
            if k in label_lower:
                return "ENGEL"
        for k in self.SLOW_KEYS:
            if k in label_lower:
                return "YAVASLA"
        for k in self.RIGHT_KEYS:
            if k in label_lower:
                return "SAGA_DON"
        for k in self.LEFT_KEYS:
            if k in label_lower:
                return "SOLA_DON"
        return "BILGI"

    def run(self):
        try:
            from ultralytics import YOLO
            print(f"[*] YOLO Modeli Yükleniyor: {self.model_path}")
            model = YOLO(self.model_path, task="detect")
            print("[+] YOLO Modeli Başarıyla Yüklendi.")
        except Exception as e:
            print(f"[-] YOLO modeli yüklenemedi: {e}")
            return

        print("[+] YOLO Nesne & Levha Algılama İş Parçacığı Aktif.")
        kare_sayaci = 0
        fps_zaman = time.time()

        while self.shared_state.running:
            start_t = time.time()
            frame = None
            with self.shared_state.lock:
                if self.shared_state.latest_frame is not None:
                    frame = self.shared_state.latest_frame.copy()

            if frame is None:
                time.sleep(0.02)
                continue

            h, w = frame.shape[:2]

            # YOLO Çıkarımı
            results = model.predict(frame, conf=self.conf_thresh, imgsz=416, verbose=False)
            
            detections = []
            emergency_brake = False
            active_sign = None
            speed_mult = 1.0

            # Güvenlik Koridoru (Tehlike Bölgesi): Alt %50 ve Orta %60
            corridor_y_min = int(h * 0.50)
            corridor_x_min = int(w * 0.20)
            corridor_x_max = int(w * 0.80)

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                names = results[0].names

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    label = names.get(cls_id, f"cls_{cls_id}")
                    label_lower = label.lower()
                    sem_type = self._sinif_tipi_belirle(label_lower)

                    detections.append({
                        'box': (x1, y1, x2, y2),
                        'label': label,
                        'conf': conf,
                        'type': sem_type
                    })

                    box_center_x = (x1 + x2) // 2

                    # 1. Engel Durumu
                    if sem_type == "ENGEL":
                        if y2 >= corridor_y_min and (corridor_x_min <= box_center_x <= corridor_x_max):
                            emergency_brake = True
                            active_sign = f"ENGEL ({label})"

                    # 2. Dur / Kırmızı Işık Levhası
                    elif sem_type == "DUR":
                        box_area = (x2 - x1) * (y2 - y1)
                        if box_area >= (w * h * 0.015):
                            active_sign = f"DUR ({label})"

                    # 3. Yavaşlama Levhaları
                    elif sem_type == "YAVASLA":
                        speed_mult = min(speed_mult, 0.6)
                        if active_sign is None:
                            active_sign = f"YAVASLA ({label})"

            kare_sayaci += 1
            if time.time() - fps_zaman >= 1.0:
                current_fps = kare_sayaci / (time.time() - fps_zaman)
                kare_sayaci = 0
                fps_zaman = time.time()
            else:
                current_fps = getattr(self, "current_fps", 18.0)
            self.current_fps = current_fps

            with self.shared_state.lock:
                self.shared_state.detections = detections
                self.shared_state.emergency_stop = emergency_brake
                self.shared_state.active_sign = active_sign
                self.shared_state.speed_multiplier = speed_mult
                self.shared_state.yolo_fps = current_fps
                if len(detections) > 0:
                    top_det = max(detections, key=lambda d: d['conf'])
                    self.shared_state.last_detected_sign = f"{top_det['label']} (%{int(top_det['conf']*100)})"
                    self.shared_state.last_detected_time = datetime.datetime.now().strftime("%H:%M:%S")

            elapsed = time.time() - start_t
            sleep_time = max(0.001, self.interval - elapsed)
            time.sleep(sleep_time)

        print("[*] YOLO iş parçacığı sonlandı.")


# --- 4. 5-INCH (800x480) RASPBERRY PI DİJİTAL KOKPİT ARAYÜZÜ ---
class Dashboard5Inch:
    """
    Raspberry Pi 5'e bağlı 5 inç ekranlar (800x480) için optimize edilmiş,
    düşük gecikmeli, modern koyu temalı kokpit ve telemetri arayüzü.
    """
    def __init__(self, width=800, height=480):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Renk Paleti (BGR)
        self.C_BG = (20, 18, 16)          # Koyu Karbon Arka Plan
        self.C_PANEL = (35, 30, 26)       # Panel Kart Rengi
        self.C_BORDER = (65, 55, 48)      # Kenarlık Rengi
        self.C_TEXT_MUTED = (160, 150, 140)
        self.C_WHITE = (245, 245, 245)
        self.C_CYAN = (235, 180, 0)       # Vurgu / Başlık
        self.C_GREEN = (60, 215, 60)      # Normal / Başarılı
        self.C_ORANGE = (0, 165, 255)     # Uyarı / Yavaşlama
        self.C_RED = (50, 50, 240)        # Acil Durum / Fren

    def draw_progress_bar(self, x, y, w, h, pct, color):
        cv2.rectangle(self.canvas, (x, y), (x + w, y + h), (45, 40, 35), -1)
        fill_w = int(w * max(0.0, min(1.0, pct / 100.0)))
        if fill_w > 0:
            cv2.rectangle(self.canvas, (x, y), (x + fill_w, y + h), color, -1)
        cv2.rectangle(self.canvas, (x, y), (x + w, y + h), (75, 65, 55), 1)

    def render(self, frame, road_mask, detections, road_detected, error,
               left_pwm, right_pwm, u_fps, y_fps, action_status,
               emergency_stop, telemetry, last_sign, last_time, arduino_connected):

        self.canvas[:] = self.C_BG

        # ==================== 1. SOL BÖLGE: KAMERA & YAPAY ZEKA GÖRSELİ (500x390) ====================
        cam_x, cam_y, cam_w, cam_h = 10, 10, 500, 390
        cv2.rectangle(self.canvas, (cam_x - 1, cam_y - 1), (cam_x + cam_w + 1, cam_y + cam_h + 1), self.C_BORDER, 1)

        if frame is not None:
            vis_cam = cv2.resize(frame, (cam_w, cam_h))
            orig_h, orig_w = frame.shape[:2]
            scale_x = cam_w / float(orig_w)
            scale_y = cam_h / float(orig_h)

            # U-Net Yol Maskesi (Yeşil Yarı Saydam Katman)
            if road_mask is not None:
                mask_scaled = cv2.resize(road_mask, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)
                green_overlay = np.zeros_like(vis_cam)
                green_overlay[:, :] = [0, 230, 0]
                masked_green = cv2.bitwise_and(green_overlay, green_overlay, mask=mask_scaled)
                vis_cam = cv2.addWeighted(vis_cam, 1.0, masked_green, 0.35, 0)

            # Güvenlik Koridoru (Tehlike Bölgesi Kutusu)
            c_y1 = int(cam_h * 0.50)
            c_x1 = int(cam_w * 0.20)
            c_x2 = int(cam_w * 0.80)
            cv2.rectangle(vis_cam, (c_x1, c_y1), (c_x2, cam_h - 4), (0, 140, 255), 1, lineType=cv2.LINE_AA)

            # YOLO Tespit Kutuları
            for det in detections:
                bx1, by1, bx2, by2 = det['box']
                sx1 = int(bx1 * scale_x)
                sy1 = int(by1 * scale_y)
                sx2 = int(bx2 * scale_x)
                sy2 = int(by2 * scale_y)

                if det['type'] in ["DUR", "ENGEL"]:
                    b_color = self.C_RED
                elif det['type'] == "YAVASLA":
                    b_color = self.C_ORANGE
                else:
                    b_color = self.C_CYAN

                cv2.rectangle(vis_cam, (sx1, sy1), (sx2, sy2), b_color, 2)
                tag = f"{det['label']} %{int(det['conf']*100)}"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(vis_cam, (sx1, max(15, sy1 - th - 4)), (sx1 + tw + 6, max(15, sy1)), b_color, -1)
                cv2.putText(vis_cam, tag, (sx1 + 3, max(13, sy1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

            # Centroid ve Hedef Çizgisi
            if road_detected:
                target_cx = int(cam_w // 2 + (error * scale_x))
                target_cx = max(10, min(cam_w - 10, target_cx))
                cv2.line(vis_cam, (cam_w // 2, cam_h - 10), (target_cx, cam_h // 2), (255, 60, 0), 2, cv2.LINE_AA)
                cv2.circle(vis_cam, (target_cx, cam_h // 2), 6, self.C_RED, -1)
                cv2.circle(vis_cam, (cam_w // 2, cam_h - 10), 4, self.C_CYAN, -1)

            # Sürüş Durumu Rozeti (Sol Üst)
            if emergency_stop or "FREN" in action_status or "DUR" in action_status:
                badge_color = self.C_RED
            elif "YAVAS" in action_status:
                badge_color = self.C_ORANGE
            else:
                badge_color = self.C_GREEN

            cv2.rectangle(vis_cam, (10, 10), (220, 36), badge_color, -1)
            cv2.putText(vis_cam, f"[>] {action_status}", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

            self.canvas[cam_y:cam_y + cam_h, cam_x:cam_x + cam_w] = vis_cam
        else:
            cv2.putText(self.canvas, "KAMERA BEKLENIYOR...", (cam_x + 100, cam_y + 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.C_TEXT_MUTED, 1)

        # ==================== 2. SAĞ BÖLGE: TELEMETRİ & SİSTEM KONTROLÜ (270x390) ====================
        p_x, p_y, p_w, p_h = 520, 10, 270, 390
        cv2.rectangle(self.canvas, (p_x, p_y), (p_x + p_w, p_y + p_h), self.C_PANEL, -1)
        cv2.rectangle(self.canvas, (p_x, p_y), (p_x + p_w, p_y + p_h), self.C_BORDER, 1)

        curr_y = p_y + 22

        # --- A. SİSTEM SAĞLIĞI (RPi 5 OS) ---
        cv2.putText(self.canvas, "SISTEM SAGLIGI (RPi 5)", (p_x + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.C_CYAN, 1, cv2.LINE_AA)
        curr_y += 6
        cv2.line(self.canvas, (p_x + 10, curr_y), (p_x + p_w - 10, curr_y), self.C_BORDER, 1)
        curr_y += 18

        # CPU Sıcaklığı
        temp_val = telemetry["cpu_temp"]
        t_color = self.C_GREEN if temp_val < 60.0 else (self.C_ORANGE if temp_val < 72.0 else self.C_RED)
        cv2.putText(self.canvas, "Islemci Sicakligi:", (p_x + 14, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(self.canvas, f"{temp_val:.1f} C", (p_x + 190, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, t_color, 1, cv2.LINE_AA)
        curr_y += 20

        # CPU Yükü
        cpu_val = telemetry["cpu_pct"]
        cv2.putText(self.canvas, f"CPU Yuku: %{cpu_val:.0f}", (p_x + 14, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_TEXT_MUTED, 1, cv2.LINE_AA)
        self.draw_progress_bar(p_x + 120, curr_y - 10, 135, 11, cpu_val, self.C_CYAN)
        curr_y += 20

        # RAM Kullanımı
        ram_pct = telemetry["ram_pct"]
        cv2.putText(self.canvas, f"RAM (%{ram_pct:.0f}):", (p_x + 14, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_TEXT_MUTED, 1, cv2.LINE_AA)
        self.draw_progress_bar(p_x + 120, curr_y - 10, 135, 11, ram_pct, self.C_GREEN)
        curr_y += 28

        # --- B. YAPAY ZEKA MODELLERİ ---
        cv2.putText(self.canvas, "YAPAY ZEKA PERFORMANSI", (p_x + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.C_CYAN, 1, cv2.LINE_AA)
        curr_y += 6
        cv2.line(self.canvas, (p_x + 10, curr_y), (p_x + p_w - 10, curr_y), self.C_BORDER, 1)
        curr_y += 18

        cv2.putText(self.canvas, "U-Net (Yol Seridi):", (p_x + 14, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(self.canvas, f"{u_fps:4.1f} FPS", (p_x + 185, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.C_GREEN, 1, cv2.LINE_AA)
        curr_y += 18

        cv2.putText(self.canvas, "YOLOv8 (Levhalar):", (p_x + 14, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(self.canvas, f"{y_fps:4.1f} FPS", (p_x + 185, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.C_CYAN, 1, cv2.LINE_AA)
        curr_y += 28

        # --- C. SÜRÜŞ & ARDUINO TELEMETRİSİ ---
        cv2.putText(self.canvas, "MOTOR VE YON KONTROLU", (p_x + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.C_CYAN, 1, cv2.LINE_AA)
        curr_y += 6
        cv2.line(self.canvas, (p_x + 10, curr_y), (p_x + p_w - 10, curr_y), self.C_BORDER, 1)
        curr_y += 18

        # Direksiyon Hatası
        err_str = f"{error:+4d} px" if road_detected else "YOL YOK"
        err_color = self.C_GREEN if abs(error) <= 25 else (self.C_ORANGE if abs(error) <= 75 else self.C_RED)
        cv2.putText(self.canvas, "Direksiyon Hatasi:", (p_x + 14, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(self.canvas, err_str, (p_x + 175, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, err_color, 1, cv2.LINE_AA)
        curr_y += 20

        # Motor PWM Barları
        cv2.putText(self.canvas, f"Sol PWM: {left_pwm:3d}", (p_x + 14, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.C_WHITE, 1, cv2.LINE_AA)
        self.draw_progress_bar(p_x + 120, curr_y - 9, 135, 9, (left_pwm / 255.0) * 100, self.C_CYAN)
        curr_y += 18

        cv2.putText(self.canvas, f"Sag PWM: {right_pwm:3d}", (p_x + 14, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.C_WHITE, 1, cv2.LINE_AA)
        self.draw_progress_bar(p_x + 120, curr_y - 9, 135, 9, (right_pwm / 255.0) * 100, self.C_CYAN)
        curr_y += 24

        # --- D. SON TESPİT EDİLEN LEVHA ---
        cv2.putText(self.canvas, "Son Tespit:", (p_x + 14, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.C_TEXT_MUTED, 1, cv2.LINE_AA)
        sign_display = last_sign[:20] if last_sign else "Bekleniyor"
        cv2.putText(self.canvas, sign_display, (p_x + 105, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.C_ORANGE, 1, cv2.LINE_AA)

        # ==================== 3. ALT DURUM ÇUBUĞU (800x60) ====================
        b_x, b_y, b_w, b_h = 10, 410, 780, 60
        cv2.rectangle(self.canvas, (b_x, b_y), (b_x + b_w, b_y + b_h), self.C_PANEL, -1)
        cv2.rectangle(self.canvas, (b_x, b_y), (b_x + b_w, b_y + b_h), self.C_BORDER, 1)

        ard_text = "[+] ARDUINO: BAGLI" if arduino_connected else "[-] ARDUINO: SIMULASYON"
        ard_color = self.C_GREEN if arduino_connected else self.C_ORANGE
        cv2.putText(self.canvas, ard_text, (b_x + 15, b_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ard_color, 1, cv2.LINE_AA)

        time_str = datetime.datetime.now().strftime("%d.%m.%Y - %H:%M:%S")
        cv2.putText(self.canvas, time_str, (b_x + 580, b_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.43, self.C_TEXT_MUTED, 1, cv2.LINE_AA)

        nav_info = f"KARAR: {action_status} | SOL: {left_pwm:3d} PWM - SAG: {right_pwm:3d} PWM"
        cv2.putText(self.canvas, nav_info, (b_x + 15, b_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, self.C_WHITE, 1, cv2.LINE_AA)

        return self.canvas


# --- 5. ANA DÖNGÜ: KARAR VERİCİ, ARDUINO VE ARAYÜZ YÖNETİMİ ---
def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Otonom Kargo Aracı - Çift Modelli Kokpit Arayüzü")
    parser.add_argument("--unet", type=str, default=os.path.join(BASE_DIR, "unet_fold1.pth"),
                        help="U-Net model dosyasının yolu (.pth veya .onnx)")
    parser.add_argument("--yolo", type=str, default=os.path.join(PROJECT_ROOT, "ciktiDosyalari", "modeller", "yolo_trafik.onnx"),
                        help="YOLO model dosyasının yolu (.onnx veya .pt)")
    parser.add_argument("--camera", type=int, default=0, help="Kamera ID (varsayılan: 0)")
    parser.add_argument("--video", type=str, default="", help="Test video veya resim dosyası yolu (Kamera yerine)")
    parser.add_argument("--serial-port", type=str, default="/dev/ttyUSB0", help="Arduino seri portu")
    parser.add_argument("--gui", action="store_true", default=True, help="5 inç Kokpit Telemetri Arayüzünü Aç (Varsayılan: Açık)")
    parser.add_argument("--no-gui", dest="gui", action="store_false", help="Arayüzü kapat, yalnızca terminal telemetrisi")
    parser.add_argument("--fullscreen", action="store_true", help="Arayüzü 5 inç ekranda tam ekran yap (Kiosk Modu)")
    args = parser.parse_args()

    # YOLO Model Kontrolü
    if not os.path.exists(args.yolo):
        alt_pt = os.path.join(PROJECT_ROOT, "ciktiDosyalari", "modeller", "yolo_trafik_best.pt")
        if os.path.exists(alt_pt):
            args.yolo = alt_pt
        else:
            args.yolo = "yolov8n.pt"
            print(f"[!] Belirtilen YOLO modeli bulunamadı, genel {args.yolo} kullanılıyor.")

    # Arduino Bağlantısı
    try:
        arduino = serial.Serial(port=args.serial_port, baudrate=9600, timeout=0.1)
        time.sleep(2)
        print(f"[+] Arduino bağlantısı kuruldu ({args.serial_port}).")
        arduino_connected = True
    except Exception as e:
        print(f"[-] Arduino bağlantı uyarısı: {e} (Simülasyon Modu)")
        arduino = None
        arduino_connected = False

    shared_state = SharedState()
    shared_state.arduino_connected = arduino_connected
    pid = PIDController(kp=0.45, ki=0.01, kd=0.15, base_speed=150)

    # Kamera / Giriş Kaynağı
    cam_src = args.video if (args.video and os.path.exists(args.video)) else args.camera
    cam_thread = CameraThread(shared_state, src=cam_src)
    unet_thread = UNetTrackingThread(shared_state, args.unet, pid, target_fps=15)
    yolo_thread = YoloDetectionThread(shared_state, args.yolo, target_fps=15)

    cam_thread.start()
    unet_thread.start()
    yolo_thread.start()

    print("=" * 65)
    print(">>> 🚀 RASPBERRY PI 5 - 5 INCH KOKPİT ARAYÜZÜ BAŞLATILDI <<<")
    print(">>> Kapatmak için klavyeden 'q' tuşuna basın veya Ctrl + C <<<")
    print("=" * 65)

    dashboard = Dashboard5Inch(width=800, height=480)
    window_name = "Otonom Kargo Araci - Kokpit"

    if args.gui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if args.fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(window_name, 800, 480)

    stop_sign_hold_until = 0.0
    islem_sayaci = 0

    try:
        while shared_state.running:
            time.sleep(0.04)  # ~25 Hz Arayüz ve Kontrol Yenileme
            islem_sayaci += 1
            now = time.time()

            with shared_state.lock:
                frame = shared_state.latest_frame.copy() if shared_state.latest_frame is not None else None
                road_mask = shared_state.road_mask.copy() if shared_state.road_mask is not None else None
                road_detected = shared_state.road_detected
                error = shared_state.error_px
                pid_left = shared_state.pid_left
                pid_right = shared_state.pid_right
                emergency_stop = shared_state.emergency_stop
                active_sign = shared_state.active_sign
                speed_mult = shared_state.speed_multiplier
                detections = list(shared_state.detections)
                u_fps = shared_state.unet_fps
                y_fps = shared_state.yolo_fps
                last_sign = shared_state.last_detected_sign
                last_time = shared_state.last_detected_time

            # --- KARAR HAKEMİ (SAFETY ARBITER) ---
            action_status = "NORMAL"

            # 1. DUR Levhası Beklemesi
            if active_sign and "DUR" in active_sign and now > stop_sign_hold_until + 6.0:
                stop_sign_hold_until = now + 3.0

            if now < stop_sign_hold_until:
                final_left, final_right = 0, 0
                action_status = f"DUR LEVHASI ({stop_sign_hold_until - now:.1f}s)"

            # 2. Acil Engel Tespiti
            elif emergency_stop:
                final_left, final_right = 0, 0
                action_status = f"ACIL FREN ({active_sign})"

            # 3. Şerit Takibi
            elif road_detected:
                final_left = int(pid_left * speed_mult)
                final_right = int(pid_right * speed_mult)
                if speed_mult < 1.0:
                    action_status = f"YAVASLA ({active_sign})"
                else:
                    action_status = "SERIT TAKIBI"
            else:
                final_left, final_right = 0, 0
                action_status = "YOL KAYIP (FREN)"

            # Arduino'ya Gönder
            cmd_packet = f"<{final_left},{final_right}>\n"
            if arduino:
                try:
                    arduino.write(cmd_packet.encode())
                except Exception:
                    pass

            # RPi 5 Sistem Verilerini Oku
            telemetry = get_rpi_telemetry()

            # Arayüz Çizimi ve Gösterimi
            if args.gui:
                dash_img = dashboard.render(
                    frame=frame,
                    road_mask=road_mask,
                    detections=detections,
                    road_detected=road_detected,
                    error=error,
                    left_pwm=final_left,
                    right_pwm=final_right,
                    u_fps=u_fps,
                    y_fps=y_fps,
                    action_status=action_status,
                    emergency_stop=emergency_stop,
                    telemetry=telemetry,
                    last_sign=last_sign,
                    last_time=last_time,
                    arduino_connected=arduino_connected
                )

                cv2.imshow(window_name, dash_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q veya ESC
                    break

            # Konsol Telemetrisi (Her 10 döngüde bir)
            if islem_sayaci % 10 == 0 and not args.gui:
                print(
                    f"[{telemetry['cpu_temp']:.1f}°C | CPU: %{telemetry['cpu_pct']:.0f}] "
                    f"U-Net: {u_fps:4.1f}FPS | YOLO: {y_fps:4.1f}FPS | "
                    f"{action_status:<20} | PWM: {final_left:3d}/{final_right:3d}"
                )

    except KeyboardInterrupt:
        print("\n[*] Kullanıcı komutuyla (Ctrl+C) durduruluyor...")
    finally:
        shared_state.running = False
        time.sleep(0.2)
        if arduino:
            try:
                arduino.write(b"<0,0>\n")
                arduino.close()
            except Exception:
                pass
        cv2.destroyAllWindows()
        print("[+] Kokpit arayüzü ve sistem güvenle kapatıldı.")


if __name__ == "__main__":
    main()
