import sys
import os
import time
import threading
import argparse

# Proje Kök Dizini ve Sanal Ortam (.venv) Kontrolü
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")

# Eğer betik .venv dışındaki bir Python yorumlayıcısıyla çağrıldıysa, otomatik olarak .venv ile yeniden başlat:
if os.path.exists(VENV_PYTHON) and os.path.realpath(sys.executable) != os.path.realpath(VENV_PYTHON):
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

import cv2
import numpy as np
import serial

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
        self.detections = []  # list of dicts: {'box': (x1, y1, x2, y2), 'label': str, 'conf': float, 'type': str}
        self.emergency_stop = False
        self.active_sign = None
        self.speed_multiplier = 1.0
        self.yolo_fps = 0.0

        # Son Nihai Gönderilen PWM
        self.final_left_pwm = 0
        self.final_right_pwm = 0
        self.action_state = "BASLATILIYOR"


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


# --- 1. THREAD: KAMERA OKUMA (ZERO BUFFER LAG) ---
class CameraThread(threading.Thread):
    def __init__(self, shared_state, src=0, width=640, height=480):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.src = src
        self.width = width
        self.height = height

    def run(self):
        print("[*] Kamera okuma iş parçacığı başlatılıyor...")
        cap = cv2.VideoCapture(self.src, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.src)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"[-] Kamera açılamadı (src={self.src})!")
            with self.shared_state.lock:
                self.shared_state.running = False
            return

        print("[+] Kamera akışı aktif.")
        while self.shared_state.running:
            ret, frame = cap.read()
            if not ret:
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
        son_zaman = time.time()
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
                current_fps = getattr(self, "current_fps", 10.0)
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

        # Sınıf Eşleştirme Sözlükleri (Genişletilebilir Sınıf Yapısı)
        self.STOP_KEYS = ["stop", "dur", "red_light", "kirmizi_isik", "no_entry", "girilmez"]
        self.SLOW_KEYS = ["yield", "yol_ver", "pedestrian", "yaya", "bump", "kasis", "warning", "dikkat", "speed"]
        self.OBSTACLE_KEYS = ["obstacle", "engel", "cone", "duba", "box", "koli", "person", "insan", "car", "araba", "bicycle", "bisiklet"]
        self.RIGHT_KEYS = ["turn_right", "sag"]
        self.LEFT_KEYS = ["turn_left", "sol"]

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

            # YOLO Tahmini (verbose=False ile terminali kirletmez)
            results = model.predict(frame, conf=self.conf_thresh, imgsz=416, verbose=False)
            
            detections = []
            emergency_brake = False
            active_sign = None
            speed_mult = 1.0

            # Güvenlik Koridoru (Tehlike Bölgesi): Aracın önü (Kameranın alt %45'lik alanı ve orta %60'ı)
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

                    # 1. Engel Durumu (Güvenlik koridorunda mı?)
                    if sem_type == "ENGEL":
                        if y2 >= corridor_y_min and (corridor_x_min <= box_center_x <= corridor_x_max):
                            emergency_brake = True
                            active_sign = f"ENGEL ({label})"

                    # 2. Dur Levhası Durumu
                    elif sem_type == "DUR":
                        box_area = (x2 - x1) * (y2 - y1)
                        # Belirli bir büyüklüğe ulaştığında (araca yaklaştığında) dur
                        if box_area >= (w * h * 0.02):
                            active_sign = "DUR LEVHASI"

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
                current_fps = getattr(self, "current_fps", 15.0)
            self.current_fps = current_fps

            with self.shared_state.lock:
                self.shared_state.detections = detections
                self.shared_state.emergency_stop = emergency_brake
                self.shared_state.active_sign = active_sign
                self.shared_state.speed_multiplier = speed_mult
                self.shared_state.yolo_fps = current_fps

            elapsed = time.time() - start_t
            sleep_time = max(0.001, self.interval - elapsed)
            time.sleep(sleep_time)

        print("[*] YOLO iş parçacığı sonlandı.")


# --- 4. ANA DÖNGÜ: KARAR VERİCİ VE ARDUINO SERİ İLETİŞİMİ ---
def main():
    parser = argparse.ArgumentParser(description="Çift Modelli Paralel Otonom Sürüş Sistemi (U-Net + YOLO)")
    parser.add_argument("--unet", type=str, default=os.path.join(BASE_DIR, "unet_fold1.pth"),
                        help="U-Net model dosyasının yolu (.pth veya .onnx)")
    parser.add_argument("--yolo", type=str, default=os.path.join(PROJECT_ROOT, "ciktiDosyalari", "modeller", "yolo_trafik.onnx"),
                        help="YOLO model dosyasının yolu (.onnx veya .pt)")
    parser.add_argument("--camera", type=int, default=0, help="Kamera ID (varsayılan: 0)")
    parser.add_argument("--serial-port", type=str, default="/dev/ttyUSB0", help="Arduino seri portu")
    parser.add_argument("--gui", action="store_true", help="Görsel telemetri penceresini aç")
    args = parser.parse_args()

    # Yedek Yol Kontrolü
    if not os.path.exists(args.yolo):
        alt_pt = os.path.join(PROJECT_ROOT, "ciktiDosyalari", "modeller", "yolo_trafik_best.pt")
        if os.path.exists(alt_pt):
            args.yolo = alt_pt
        else:
            args.yolo = "yolov8n.pt"  # Otomatik fallback
            print(f"[!] Belirtilen YOLO modeli bulunamadı, varsayılan {args.yolo} kullanılıyor.")

    # Arduino Bağlantısı
    try:
        arduino = serial.Serial(port=args.serial_port, baudrate=9600, timeout=0.1)
        time.sleep(2)
        print(f"[+] Arduino bağlantısı kuruldu ({args.serial_port}).")
    except Exception as e:
        print(f"[-] Arduino bağlantı uyarısı: {e} (Simülasyon Modu)")
        arduino = None

    shared_state = SharedState()
    pid = PIDController(kp=0.45, ki=0.01, kd=0.15, base_speed=150)

    # İş Parçacıklarını Başlat
    cam_thread = CameraThread(shared_state, src=args.camera)
    unet_thread = UNetTrackingThread(shared_state, args.unet, pid, target_fps=15)
    yolo_thread = YoloDetectionThread(shared_state, args.yolo, target_fps=15)

    cam_thread.start()
    unet_thread.start()
    yolo_thread.start()

    print("=" * 65)
    print(">>> 🚀 ÇİFT MODELLİ OTONOM SÜRÜŞ SİSTEMİ AKTİF <<<")
    print(">>> Durdurmak için: Ctrl + C <<<")
    print("=" * 65)

    stop_sign_hold_until = 0.0
    islem_sayaci = 0

    try:
        while shared_state.running:
            time.sleep(0.05)  # 20 Hz Kontrol Döngüsü
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

            if frame is None:
                continue

            # --- KARAR HAKEMİ (SAFETY ARBITER) ---
            action_status = "NORMAL"

            # 1. Öncelik: DUR Levhası Bekleme Sayacı
            if active_sign == "DUR LEVHASI" and now > stop_sign_hold_until + 6.0:
                stop_sign_hold_until = now + 3.0  # 3 saniye dur

            if now < stop_sign_hold_until:
                final_left, final_right = 0, 0
                action_status = f"BEKLEME (DUR LEVHASI: {stop_sign_hold_until - now:.1f}s)"

            # 2. Öncelik: Acil Engel Tespiti
            elif emergency_stop:
                final_left, final_right = 0, 0
                action_status = f"ACIL FREN ({active_sign})"

            # 3. Öncelik: Yol Takibi
            elif road_detected:
                final_left = int(pid_left * speed_mult)
                final_right = int(pid_right * speed_mult)
                if speed_mult < 1.0:
                    action_status = f"YAVAS SÜRÜS ({active_sign})"
                else:
                    action_status = "SERIT TAKIBI"
            else:
                final_left, final_right = 0, 0
                action_status = "YOL KAYIP (FREN)"

            # Arduino'ya Gönder
            cmd_packet = f"<{final_left},{final_right}>\n"
            if arduino:
                arduino.write(cmd_packet.encode())

            # Konsol Telemetrisi (Her 4 döngüde bir)
            if islem_sayaci % 4 == 0:
                print(
                    f"[#{islem_sayaci:04d}] U-Net: {u_fps:4.1f}FPS | YOLO: {y_fps:4.1f}FPS | "
                    f"Durum: {action_status:<22} | PWM: {final_left:3d}/{final_right:3d}"
                )

            # Görselleştirme (GUI Modu İstenirse)
            if args.gui:
                vis_frame = frame.copy()
                if road_mask is not None:
                    green = np.zeros_like(vis_frame)
                    green[:, :] = [0, 255, 0]
                    road_overlay = cv2.bitwise_and(green, green, mask=road_mask)
                    vis_frame = cv2.addWeighted(vis_frame, 1.0, road_overlay, 0.35, 0)

                for det in detections:
                    bx1, by1, bx2, by2 = det['box']
                    color = (0, 0, 255) if det['type'] in ["DUR", "ENGEL"] else (0, 255, 255)
                    cv2.rectangle(vis_frame, (bx1, by1), (bx2, by2), color, 2)
                    label_str = f"{det['label']} {det['conf']:.2f}"
                    cv2.putText(vis_frame, label_str, (bx1, max(20, by1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.putText(vis_frame, f"STATUS: {action_status}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(vis_frame, f"U-Net: {u_fps:.1f} FPS | YOLO: {y_fps:.1f} FPS", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow("Otonom Kargo Araci - Cift Modelli Telemetri", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n[*] Kullanıcı komutuyla durduruluyor...")
    finally:
        shared_state.running = False
        time.sleep(0.2)
        if arduino:
            arduino.write(b"<0,0>\n")
            arduino.close()
        cv2.destroyAllWindows()
        print("[+] Tüm bağlantılar ve modeller güvenle kapatıldı.")


if __name__ == "__main__":
    main()
