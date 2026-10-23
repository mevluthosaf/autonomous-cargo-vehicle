# 🚚 Autonomous Cargo Delivery Vehicle

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX%20Runtime-grey.svg?logo=onnx)](https://onnxruntime.ai/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg?logo=opencv)](https://opencv.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205%20%7C%20Arduino%20Uno-red.svg)](https://www.raspberrypi.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An autonomous carrier platform prototype developed to reduce human labor and minimize routing errors in urban logistics by **real-time segmentation and tracking of dedicated bike paths using deep learning-based computer vision algorithms**.

---

## 📌 Table of Contents
- [Key Features & Objectives](#-key-features--objectives)
- [System & Software Architecture](#-system--software-architecture)
- [Hardware & Power Architecture](#-hardware--power-architecture)
- [Algorithm & Decision-Making Pipeline](#-algorithm--decision-making-pipeline)
- [Installation & Getting Started](#-installation--getting-started)
- [Live Field Tests & Results](#-live-field-tests--results)
- [Project Team & Supervisor](#-project-team--supervisor)

---

## 🎯 Key Features & Objectives

* **Custom Dataset & Semantic Segmentation:** Utilizes a pixel-level **U-Net** semantic segmentation architecture to overcome the limitations of traditional bounding-box object detection methods on continuous path boundaries.
* **Edge AI Optimization:** The trained deep learning model is converted to **ONNX** format and executed via ONNX Runtime for low-latency inference on the Raspberry Pi 5.
* **Non-blocking Multithreading:** Video frame capture, neural network inference, and serial port communication run concurrently on separate threads to eliminate performance bottlenecks.
* **Dual-Rail Isolated Power Distribution:** Engineered with dual XL4016 buck converters to prevent motor current surges from causing voltage drops and resetting the onboard computer.

---

## 🧠 System & Software Architecture

### 1. U-Net Semantic Segmentation Model
A 23-layer convolutional neural network with Contracting (Encoder) and Expanding (Decoder) paths joined via skip connections is employed for sharp, continuous path masking. The model’s generalization capacity was verified using **5-Fold Cross-Validation**.

| Parameter | Specification / Description |
| :--- | :--- |
| **Input Resolution** | 640x640 / Single Channel (Grayscale) |
| **Data Augmentation** | Brightness, Exposure, Rotation, Blur & Noise injection |
| **Runtime Engine** | ONNX Runtime (Embedded Edge Optimization) |

---

## 🔌 Hardware & Power Architecture

System tasks are distributed across two processing units to ensure high compute efficiency and reliable low-level control:

* **Raspberry Pi 5 (8 GB RAM):** High-level decision making, camera stream management, deep learning inference, and path deviation calculations.
* **Arduino Uno:** Low-level hardware control, HC-SR04 ultrasonic distance sensor readout, and motor PWM signal generation.
* **L298N Motor Driver & 4x 6V DC Motors:** Differential maneuverability based on the **skid-steer (tank steering)** mechanism.
* **Logitech C270 Webcam:** 720p real-time environmental visual input.

```text
               ┌────────────────────────────────────────────────────────┐
               │            Jetfire 14.8V 5200 mAh Li-Po Battery        │
               └───────────────────┬────────────────────────────────┬───┘
                                   │                                │
                                   ▼                                ▼
                    ┌────────────────────────────┐    ┌────────────────────────────┐
                    │ XL4016 Buck Converter #1   │    │ XL4016 Buck Converter #2   │
                    │  (14.8V -> 5.1V Regulated) │    │  (14.8V -> 9.1V Regulated) │
                    └──────────────┬─────────────┘    └─────────────┬──────────────┘
                                   │                                │
                                   ▼ (Type-C)                       ▼
                         ┌───────────────────┐            ┌───────────────────┐
                         │  Raspberry Pi 5   │            │  L298N Driver &   │
                         │   (8GB LPDDR4X)   │            │    4x DC Motors   │
                         └─────────┬─────────┘            └─────────▲─────────┘
                                   │                                │ (PWM)
                                   │ (USB Serial - UART)            │
                                   ▼                                │
                         ┌──────────────────────────────────────────┴┐
                         │               Arduino Uno                 │
                         │      (HC-SR04 Ultrasonic Sensor)          │
                         └───────────────────────────────────────────┘
```

---

## 🔄 Algorithm & Decision-Making Pipeline

1. **Frame Capture:** The real-time camera frame is preprocessed (Auto-Orient, Resize, Grayscale) and passed to the ONNX model.
2. **Binary Mask Generation:** The U-Net output is converted to a binary path mask.
3. **Centroid Calculation:** OpenCV `moments` calculate the center of mass of the segmented path in real time.
4. **Deviation Analysis:** The pixel offset between the camera's center axis and the path centroid determines the trajectory error. If this error exceeds the defined threshold, a steering command (left/right skid-steer) is issued.
5. **Serial Execution:** Driving commands are transmitted via USB-UART to the Arduino Uno to adjust motor PWM states.

---

## 🚀 Installation & Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/mevluthosaf/autonomous-cargo-vehicle.git](https://github.com/mevluthosaf/autonomous-cargo-vehicle.git)
cd autonomous-cargo-vehicle
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Flash the Arduino Firmware
Upload `arduino/motor_and_sensors.ino` to the Arduino Uno board using the Arduino IDE.

### 4. Run the Main Controller
```bash
python ana_program/main.py
```

---

## 📸 Live Field Tests & Results

| Real-Time U-Net Segmentation Output | Autonomous Driving Field Test |
| :---: | :---: |
| ![Segmentation Output](assets/segmentation_live.png) | ![Field Test](assets/field_test.png) |

| Prototype Front View | Prototype Side View |
| :---: | :---: |
| ![Front View](assets/vehicle_front.png) | ![Side View](assets/vehicle_side.png) |

---

## 👥 Project Team & Supervisor

* **Developers:** Zeliha Önel & Mevlüt Hoşaf & Burak Güdücü
* **Academic Advisor:** Assoc. Prof. Dr. Ömer Kaan Baykan
* **Institution:** Konya Technical University — Department of Computer Engineering
