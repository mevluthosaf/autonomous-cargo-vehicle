import sys
import os
import argparse

# Proje Kök Dizini (autonomous-cargo-vehicle/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(BASE_DIR, ".venv", "bin", "python")

# Eğer betik .venv dışındaki bir Python yorumlayıcısıyla çağrıldıysa, otomatik olarak .venv ile yeniden başlat:
if os.path.exists(VENV_PYTHON) and os.path.realpath(sys.executable) != os.path.realpath(VENV_PYTHON):
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from ultralytics import YOLO
from tqdm import tqdm

CIHAZ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VARSAYILAN_VIDEO = os.path.join(BASE_DIR, "assets", "test_video.mp4")
CIKTI_KLASORU = os.path.join(BASE_DIR, "ciktiDosyalari", "test_videolari")
UNET_MODEL = os.path.join(BASE_DIR, "ana_program", "unet_fold1.pth")
YOLO_MODEL = os.path.join(BASE_DIR, "ciktiDosyalari", "modeller", "yolo_trafik_best.pt")


def main():
    parser = argparse.ArgumentParser(description="U-Net ve YOLO Modellerini Birlikte Video Üzerinde Test Etme")
    parser.add_argument("--video", type=str, default=VARSAYILAN_VIDEO, help="Test videosunun yolu")
    parser.add_argument("--unet", type=str, default=UNET_MODEL, help="U-Net model ağırlığı (.pth)")
    parser.add_argument("--yolo", type=str, default=YOLO_MODEL, help="YOLO model ağırlığı (.pt veya .onnx)")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO tespit güven eşiği")
    parser.add_argument("--view", action="store_true", help="İşleme anında pencerede canlı izle")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[-] Video bulunamadı: {args.video}")
        return

    os.makedirs(CIKTI_KLASORU, exist_ok=True)

    # 1. U-Net Modeli Yükle
    print(f"[*] U-Net Modeli Yükleniyor ({CIHAZ}): {args.unet}")
    unet_model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1).to(CIHAZ)
    if os.path.exists(args.unet):
        unet_model.load_state_dict(torch.load(args.unet, map_location=CIHAZ))
        unet_model.eval()
        print("[+] U-Net yüklendi.")
    else:
        print(f"[-] U-Net dosyası bulunamadı ({args.unet}), sadece YOLO çalışacak.")
        unet_model = None

    # 2. YOLO Modeli Yükle
    yolo_path = args.yolo
    if not os.path.exists(yolo_path):
        alt_onnx = os.path.join(BASE_DIR, "ciktiDosyalari", "modeller", "yolo_trafik.onnx")
        if os.path.exists(alt_onnx):
            yolo_path = alt_onnx
        else:
            yolo_path = "yolov8n.pt"  # Pre-trained fallback
            print(f"[!] Eğitilmiş YOLO bulunamadı, genel nesneler için {yolo_path} kullanılıyor.")

    print(f"[*] YOLO Modeli Yükleniyor: {yolo_path}")
    yolo_model = YOLO(yolo_path)
    print("[+] YOLO yüklendi.")

    # 3. Videoyu Aç
    cap = cv2.VideoCapture(args.video)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    toplam_kare = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cikti_video_yolu = os.path.join(CIKTI_KLASORU, "cift_model_test_cikti.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(cikti_video_yolu, fourcc, fps, (width, height))

    print(f"\n[*] Video işleniyor: {args.video} ({toplam_kare} kare, {fps} FPS)...")

    for _ in tqdm(range(toplam_kare), desc="İşleniyor"):
        ret, frame = cap.read()
        if not ret:
            break

        vis_frame = frame.copy()

        # --- A. U-Net Şerit Maskesi ---
        if unet_model is not None:
            resim_resized = cv2.resize(frame, (640, 640))
            resim_rgb = cv2.cvtColor(resim_resized, cv2.COLOR_BGR2RGB)
            input_tensor = torch.from_numpy(resim_rgb.transpose(2, 0, 1)).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0).to(CIHAZ)

            with torch.no_grad():
                cikti = unet_model(input_tensor)
                maske = (torch.sigmoid(cikti).squeeze().cpu().numpy() > 0.70).astype(np.uint8) * 255

            mask_orig = cv2.resize(maske, (width, height), interpolation=cv2.INTER_NEAREST)
            yesil_katman = np.zeros_like(frame)
            yesil_katman[:, :] = [0, 255, 0]
            boyali_alan = cv2.bitwise_and(yesil_katman, yesil_katman, mask=mask_orig)
            vis_frame = cv2.addWeighted(vis_frame, 1.0, boyali_alan, 0.35, 0)

            # Centroid
            M = cv2.moments(mask_orig)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(vis_frame, (cx, cy), 8, (0, 0, 255), -1)
                cv2.line(vis_frame, (width // 2, height), (cx, cy), (255, 0, 0), 2)

        # --- B. YOLO Nesne ve Levha Tespiti ---
        results = yolo_model.predict(frame, conf=args.conf, imgsz=640, verbose=False)
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = results[0].names.get(cls_id, f"cls_{cls_id}")

                color = (0, 140, 255) if "stop" in label.lower() or "dur" in label.lower() else (255, 255, 0)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_frame, f"{label} %{int(conf*100)}", (x1, max(20, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(vis_frame)

        if args.view:
            cv2.imshow("Cift Modelli Test", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n[+] İşlem tamamlandı! Çıktı videosu: {cikti_video_yolu}")


if __name__ == "__main__":
    main()
