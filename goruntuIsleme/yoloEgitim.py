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
import yaml
from pathlib import Path

# --- KULLANICI / EĞİTİM AYARLARI ---
VARSAYILAN_EPOCH = 50
VARSAYILAN_BATCH = 16
VARSAYILAN_IMGSZ = 640
CIKTI_KLASORU = os.path.join(BASE_DIR, "ciktiDosyalari", "modeller")
DATASET_KLASORU = os.path.join(BASE_DIR, "dataset_yolo")


def roboflow_veri_seti_indir(api_key, workspace, project, version):
    """
    Roboflow API kullanarak veri setini otomatik olarak indirir ve data.yaml yolunu döner.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[-] 'roboflow' kütüphanesi yüklü değil. 'pip install roboflow' çalıştırın.")
        sys.exit(1)

    print(f"\n[*] Roboflow'dan veri seti çekiliyor: {workspace}/{project} (v{version})...")
    rf = Roboflow(api_key=api_key)
    rf_project = rf.workspace(workspace).project(project)
    rf_dataset = rf_project.version(version).download("yolov8", location=DATASET_KLASORU)
    
    yaml_yolu = os.path.join(rf_dataset.location, "data.yaml")
    print(f"[+] Veri seti başarıyla indirildi: {yaml_yolu}")
    return yaml_yolu


def yolo_egit_ve_donustur(data_yaml_yolu, model_adi="yolov8n.pt", epochs=50, batch=16, imgsz=640):
    """
    YOLO modelini eğitir, değerlendirir ve Raspberry Pi için ONNX formatına dönüştürür.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[-] 'ultralytics' kütüphanesi yüklü değil. 'pip install ultralytics' çalıştırın.")
        sys.exit(1)

    os.makedirs(CIKTI_KLASORU, exist_ok=True)

    # 1. data.yaml içeriğini ve sınıfları doğrula
    if not os.path.exists(data_yaml_yolu):
        print(f"[-] Hata: {data_yaml_yolu} bulunamadı!")
        sys.exit(1)

    with open(data_yaml_yolu, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    siniflar = data_cfg.get("names", [])
    sinif_sayisi = len(siniflar) if isinstance(siniflar, list) else len(siniflar.keys())
    print("=" * 60)
    print(f"[*] YOLO Modeli Eğitimi Başlatılıyor...")
    print(f"[*] Model Tabanı   : {model_adi}")
    print(f"[*] Sınıf Sayısı   : {sinif_sayisi}")
    print(f"[*] Tanımlı Sınıflar: {siniflar}")
    print(f"[*] Giriş Çözünürlüğü: {imgsz}x{imgsz} | Epochs: {epochs} | Batch: {batch}")
    print("=" * 60)

    # 2. Modeli Başlat (Pre-trained weights)
    model = YOLO(model_adi)

    # 3. Eğitimi Başlat
    proje_cikti_yolu = os.path.join(BASE_DIR, "ciktiDosyalari", "yolo_egitim_kayitlari")
    sonuclar = model.train(
        data=data_yaml_yolu,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=proje_cikti_yolu,
        name="trafik_isaretleri",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        save=True
    )

    best_pt_yolu = os.path.join(proje_cikti_yolu, "trafik_isaretleri", "weights", "best.pt")
    hedef_pt_yolu = os.path.join(CIKTI_KLASORU, "yolo_trafik_best.pt")
    
    if os.path.exists(best_pt_yolu):
        import shutil
        shutil.copy2(best_pt_yolu, hedef_pt_yolu)
        print(f"\n[+] En iyi model ağırlığı kaydedildi: {hedef_pt_yolu}")
        egitilmis_model = YOLO(hedef_pt_yolu)
    else:
        egitilmis_model = model

    # 4. Modeli Değerlendir (Validation)
    print("\n[*] Model Doğrulama (Validation) Metrikleri Hesaplanıyor...")
    val_metrics = egitilmis_model.val()
    print(f"[+] mAP50: {val_metrics.box.map50:.4f} | mAP50-95: {val_metrics.box.map:.4f}")

    # 5. Raspberry Pi İçin ONNX Formatına Dönüştür (Edge Optimization)
    print("\n[*] Raspberry Pi 5 için ONNX formatına dönüştürülüyor (dynamic=False, opset=12)...")
    onnx_yolu = egitilmis_model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        opset=12,
        simplify=True
    )
    
    # ONNX modelini çıktılar klasörüne kopyala
    hedef_onnx = os.path.join(CIKTI_KLASORU, "yolo_trafik.onnx")
    if os.path.exists(onnx_yolu) and onnx_yolu != hedef_onnx:
        import shutil
        shutil.copy2(onnx_yolu, hedef_onnx)
        print(f"[+] ONNX modeli hazır: {hedef_onnx}")
    else:
        print(f"[+] ONNX modeli hazır: {onnx_yolu}")

    print("\n" + "=" * 60)
    print("🎉 Eğitim ve Optimizasyon Başarıyla Tamamlandı!")
    print(f"-> PyTorch Modeli: {hedef_pt_yolu}")
    print(f"-> RPi ONNX Modeli: {hedef_onnx}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roboflow Destekli YOLO Trafik İşaretleri ve Engel Eğitimi")
    
    # Roboflow İndirme Parametreleri (İsteğe Bağlı)
    parser.add_argument("--roboflow", action="store_true", help="Veri setini Roboflow'dan otomatik indir")
    parser.add_argument("--api-key", type=str, default="", help="Roboflow API Anahtarı")
    parser.add_argument("--workspace", type=str, default="", help="Roboflow Workspace Adı")
    parser.add_argument("--project", type=str, default="", help="Roboflow Proje Adı")
    parser.add_argument("--version", type=int, default=1, help="Roboflow Veri Seti Sürümü")

    # Yerel Veri Seti Parametresi
    parser.add_argument("--data", type=str, default=os.path.join(DATASET_KLASORU, "data.yaml"),
                        help="data.yaml dosyasının tam yolu")

    # Eğitim Hiperparametreleri
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Başlangıç ağırlığı (yolov8n.pt, yolo11n.pt vb.)")
    parser.add_argument("--epochs", type=int, default=VARSAYILAN_EPOCH, help="Epoch sayısı")
    parser.add_argument("--batch", type=int, default=VARSAYILAN_BATCH, help="Batch boyutu")
    parser.add_argument("--imgsz", type=int, default=VARSAYILAN_IMGSZ, help="Giriş resim boyutu (örn: 640 veya 416)")

    args = parser.parse_args()

    # 1. Veri seti temini
    if args.roboflow:
        if not args.api_key or not args.workspace or not args.project:
            print("[-] Hata: Roboflow indirme için --api-key, --workspace ve --project parametreleri zorunludur.")
            sys.exit(1)
        data_yaml = roboflow_veri_seti_indir(args.api_key, args.workspace, args.project, args.version)
    else:
        data_yaml = args.data
        if not os.path.exists(data_yaml):
            print(f"[-] Belirtilen data.yaml bulunamadı: {data_yaml}")
            print("[*] Eğer veri setini Roboflow'dan indirecekseniz:")
            print("    python goruntuIsleme/yoloEgitim.py --roboflow --api-key YOUR_KEY --workspace WORKSPACE --project PROJECT --version 1")
            print("[*] Veya indirilen klasörün data.yaml yolunu verin:")
            print("    python goruntuIsleme/yoloEgitim.py --data /yol/dataset/data.yaml")
            sys.exit(1)

    # 2. Modeli eğit ve ONNX'e dönüştür
    yolo_egit_ve_donustur(
        data_yaml_yolu=data_yaml,
        model_adi=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz
    )
