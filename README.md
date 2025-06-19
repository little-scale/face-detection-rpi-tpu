# 🧠 Coral EdgeTPU Face Detection on Raspberry Pi 4 (Headless)

This project enables real-time **face detection** on a **headless Raspberry Pi 4** using a **USB webcam** and a **Google Coral USB Accelerator**. It leverages a quantized **SSD MobileNet V2** model optimized for EdgeTPU, and performs inference with the `pycoral` library.

---

## 📋 Overview

- 🐍 Python 3.7 (built from source)
- ⚙️ Coral EdgeTPU USB Accelerator
- 📷 Headless USB webcam input via OpenCV
- 📦 Dependencies: `tflite-runtime`, `pycoral`, `opencv-python`
- ⚡ Fast inference: \~20 FPS

---

## 🪛 1. Install Raspberry Pi OS and Prepare System

Install **Raspberry Pi OS Lite (64-bit, Bookworm)** from:\
➡️ [https://www.raspberrypi.com/software/operating-systems/](https://www.raspberrypi.com/software/operating-systems/)

Enable SSH, connect to Wi-Fi, then run:

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y \
  build-essential libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev libncursesw5-dev \
  xz-utils tk-dev libffi-dev liblzma-dev \
  libgdbm-dev libnss3-dev libgdbm-compat-dev \
  libjpeg-dev libatlas-base-dev libportaudio2 \
  libopenblas-dev libhdf5-dev libblas-dev \
  liblapack-dev libudev-dev cmake pkg-config \
  libavcodec-dev libavformat-dev libswscale-dev \
  libgtk-3-dev libv4l-dev libcanberra-gtk* \
  libtbb2 libtbb-dev libdc1394-22-dev \
  usbutils wget git python3-pip
```

---

## 🐍 2. Build Python 3.7 and Set Up Virtual Environment

```bash
# Download and build Python 3.7.17
cd ~
wget https://www.python.org/ftp/python/3.7.17/Python-3.7.17.tgz
tar -xf Python-3.7.17.tgz
cd Python-3.7.17
./configure --enable-optimizations
make -j4
sudo make altinstall

# Create and activate virtual environment
cd ~
python3.7 -m venv mpenv37
source mpenv37/bin/activate
```

---

## 📦 3. Install Dependencies in Virtual Environment

```bash
# Upgrade pip and tooling
pip install --upgrade pip setuptools wheel

# Download compatible packages manually
wget https://github.com/google-coral/tflite-runtime/releases/download/v2.5.0/tflite_runtime-2.5.0-cp37-cp37m-linux_aarch64.whl
wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl

# Install them
pip install tflite_runtime-2.5.0-cp37-cp37m-linux_aarch64.whl
pip install pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl

# Install OpenCV (v4.x)
pip install opencv-python==4.5.5.64
```

---

## 📥 4. Download Model

```bash
wget https://github.com/shawwn/edge-models/raw/main/face_edgetpu.tflite -O face_edgetpu.tflite
```

---

## 🧪 5. Run Inference Script

Create a file called `face_detect_coral.py` and run it using:

```bash
python face_detect_coral.py
```

This script captures frames from the USB camera, resizes and processes them for inference on the Coral USB Accelerator, and prints the number of faces detected with their confidence scores.

You can also modify the script to save frames or export bounding box data as needed.

