import sys
import os

# Proje Kök Dizini ve Sanal Ortam (.venv) Kontrolü
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(BASE_DIR, ".venv", "bin", "python")

# Eğer betik .venv dışındaki bir Python yorumlayıcısıyla çağrıldıysa, otomatik olarak .venv ile yeniden başlat:
if os.path.exists(VENV_PYTHON) and os.path.realpath(sys.executable) != os.path.realpath(VENV_PYTHON):
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

import warnings
warnings.filterwarnings("ignore")

import argparse
import cv2
from ultralytics import YOLO

VARSAYILAN_MODEL_PT = os.path.join(BASE_DIR, "ciktiDosyalari", "modeller", "yolo_trafik_best.pt")
VARSAYILAN_MODEL_ONNX = os.path.join(BASE_DIR, "ciktiDosyalari", "modeller", "yolo_trafik.onnx")
CIKTI_KLASORU = os.path.join(BASE_DIR, "ciktiDosyalari", "test_gorselleri")


def model_sec(model_yolu):
    """Belirtilen YOLO model yolunu doğrular; yoksa mevcut yedek modelleri arar."""
    if model_yolu and os.path.exists(model_yolu):
        return model_yolu
    if os.path.exists(VARSAYILAN_MODEL_PT):
        return VARSAYILAN_MODEL_PT
    if os.path.exists(VARSAYILAN_MODEL_ONNX):
        return VARSAYILAN_MODEL_ONNX
    return "yolov8n.pt"


def gorsel_test_et(image_path, model_path=None, conf_thresh=0.45, imgsz=640, output_path=None, show_view=False):
    """Tek bir görsel üzerinde YOLO çıkarımı yapar, tespitleri yazdırır ve çıktıyı kaydeder."""
    if not os.path.exists(image_path):
        print(f"[-] Hata: Görsel dosyası bulunamadı -> {image_path}")
        return None

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[-] Hata: Görsel okunamadı (geçersiz format olabilir) -> {image_path}")
        return None

    secilen_model = model_sec(model_path)
    print(f"[*] YOLO Modeli Yükleniyor: {secilen_model}")
    model = YOLO(secilen_model, task="detect")

    print(f"[*] Görsel İşleniyor: {image_path} (conf={conf_thresh}, imgsz={imgsz})")
    results = model.predict(frame, conf=conf_thresh, imgsz=imgsz, verbose=False)

    vis_frame = frame.copy()
    algilananlar = []

    if len(results) > 0 and results[0].boxes is not None:
        names = results[0].names
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            label = names.get(cls_id, f"cls_{cls_id}")

            algilananlar.append({
                "label": label,
                "conf": conf,
                "box": (x1, y1, x2, y2),
            })

            # Renk seçimi: Dur/Kırmızı Işık -> Kırmızı, Uyarı/Sarı -> Turuncu, Diğer -> Camgöbeği
            label_lower = label.lower()
            if any(k in label_lower for k in ["stop", "dur", "red_light", "do_not_enter"]):
                color = (0, 0, 255)
            elif any(k in label_lower for k in ["warning", "yellow_light", "ped_", "yield"]):
                color = (0, 165, 255)
            else:
                color = (255, 220, 0)

            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            tag = f"{label} %{int(conf * 100)}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            tag_y = max(th + 8, y1)
            cv2.rectangle(vis_frame, (x1, tag_y - th - 8), (x1 + tw + 8, tag_y), color, -1)
            cv2.putText(vis_frame, tag, (x1 + 4, tag_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    print("\n" + "=" * 50)
    if algilananlar:
        print(f"[+] Toplam {len(algilananlar)} nesne algılandı:")
        for i, det in enumerate(algilananlar, start=1):
            bx1, by1, bx2, by2 = det["box"]
            print(f"  {i}. {det['label']:<20} | Güven: %{det['conf'] * 100:5.1f} | Kutu: ({bx1}, {by1}) -> ({bx2}, {by2})")
    else:
        print("[!] Görselde eşik değerinin üzerinde nesne algılanamadı.")
    print("=" * 50)

    # Çıktıyı kaydet
    if output_path:
        hedef_yol = output_path
        os.makedirs(os.path.dirname(os.path.abspath(hedef_yol)), exist_ok=True)
    else:
        os.makedirs(CIKTI_KLASORU, exist_ok=True)
        dosya_adi, uzanti = os.path.splitext(os.path.basename(image_path))
        if not uzanti:
            uzanti = ".jpg"
        hedef_yol = os.path.join(CIKTI_KLASORU, f"{dosya_adi}_yolo{uzanti}")

    cv2.imwrite(hedef_yol, vis_frame)
    print(f"[+] Çıktı görseli kaydedildi: {hedef_yol}")

    if show_view:
        cv2.imshow("YOLO Gorsel Test Sonucu", vis_frame)
        print("[*] Pencereyi kapatmak için herhangi bir tuşa basın...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return hedef_yol


def main():
    parser = argparse.ArgumentParser(description="YOLO Modelini Tek Bir Görsel Üzerinde Test Etme")
    parser.add_argument("gorsel", nargs="?", default="", help="Test edilecek görselin yolu")
    parser.add_argument("--image", "-i", type=str, default="", help="Test edilecek görselin yolu (alternatif)")
    parser.add_argument("--model", "-m", type=str, default=VARSAYILAN_MODEL_PT, help="YOLO model yolu (.pt veya .onnx)")
    parser.add_argument("--conf", "-c", type=float, default=0.45, help="Güven eşiği (varsayılan: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="Giriş çözünürlüğü (varsayılan: 640)")
    parser.add_argument("--output", "-o", type=str, default="", help="Çıktı görselinin kaydedileceği yol")
    parser.add_argument("--view", "-v", action="store_true", help="Çıktı görselini pencerede göster")
    args = parser.parse_args()

    image_path = args.image or args.gorsel
    if not image_path:
        image_path = input("Test edilecek görselin yolunu girin: ").strip().strip("'\"")

    gorsel_test_et(
        image_path=image_path,
        model_path=args.model,
        conf_thresh=args.conf,
        imgsz=args.imgsz,
        output_path=args.output or None,
        show_view=args.view,
    )


if __name__ == "__main__":
    main()

