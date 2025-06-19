# face-detection-rpi-tpu

🧠 Coral EdgeTPU Face Detection on Raspberry Pi 4 (Headless)
This project runs real-time face detection on a Raspberry Pi 4 using a USB webcam and the Coral USB Accelerator. Detection is done using a quantized SSD MobilenetV2 face model compiled for EdgeTPU. The script runs headless (no GUI) and prints detected faces and inference times to the terminal.

🪛 1. Raspberry Pi OS Setup
Use Raspberry Pi OS Lite 64-bit (Bookworm) from https://www.raspberrypi.com/software/operating-systems/.

Flash the image to SD card using Raspberry Pi Imager.

Enable SSH, configure WiFi, and boot the Pi.

Then:

bash
Copy
Edit
sudo apt update
sudo apt upgrade
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
🐍 2. Build and Use Python 3.7 Virtual Environment
MediaPipe + Coral requires Python 3.7.

bash
Copy
Edit
# Download and build Python 3.7
cd ~
wget https://www.python.org/ftp/python/3.7.17/Python-3.7.17.tgz
tar -xf Python-3.7.17.tgz
cd Python-3.7.17
./configure --enable-optimizations
make -j4
sudo make altinstall

# Create venv
cd ~
python3.7 -m venv mpenv37
source mpenv37/bin/activate
📦 3. Install Required Packages in Virtual Env
With the venv activated:

bash
Copy
Edit
# Install pip wheel tooling
pip install --upgrade pip setuptools wheel

# Install pycoral + tflite_runtime manually (no 3.7 pip releases)
# Download pycoral 2.0.0 and tflite-runtime manually
wget https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl
wget https://github.com/google-coral/tflite-runtime/releases/download/v2.5.0/tflite_runtime-2.5.0-cp37-cp37m-linux_aarch64.whl

# Install them
pip install tflite_runtime-2.5.0-cp37-cp37m-linux_aarch64.whl
pip install pycoral-2.0.0-cp37-cp37m-linux_aarch64.whl

# OpenCV for webcam capture
pip install opencv-python==4.5.5.64
⚙️ 4. Install EdgeTPU Compiler (Optional, for model compilation)
bash
Copy
Edit
# Add Coral repo
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update

# OR install manually:
wget https://github.com/google-coral/edgetpu/releases/download/release-dryrun-2023-03-29/edgetpu-compiler_15.0.340273435_arm64.deb
sudo dpkg -i edgetpu-compiler_15.0.340273435_arm64.deb
📥 5. Download Face Detection Model
Download precompiled model:

bash
Copy
Edit
wget https://github.com/shawwn/edge-models/raw/main/face_edgetpu.tflite -O face_edgetpu.tflite
This is a quantized SSD MobilenetV2 model for face detection, compiled for EdgeTPU.

🧪 6. Run Face Detection Script
Create and run the following script:

face_detect_coral.py
python
Copy
Edit
import time
import cv2
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect

MODEL_PATH = 'face_edgetpu.tflite'
SCORE_THRESHOLD = 0.6
FLIP_VERTICAL = True

print(f"🎯 Loading model: {MODEL_PATH}")
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()
input_size = common.input_size(interpreter)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Failed to open webcam")

print("🚀 Running Coral TPU face detection...")
frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Skipped frame")
            continue

        if FLIP_VERTICAL:
            frame = cv2.flip(frame, 0)

        resized = cv2.resize(frame, input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        common.set_input(interpreter, rgb)
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()

        objs = detect.get_objects(interpreter, SCORE_THRESHOLD)

        print(f"\n📷 Frame shape: {frame.shape}")
        print(f"🎯 Model input size: {input_size}")
        print(f"🧠 {len(objs)} face(s) detected")
        for i, o in enumerate(objs):
            print(f"  • Face {i+1}: Score={o.score:.2f}, BBox={o.bbox}")
        print(f"⏱️ Inference time: {(t1 - t0) * 1000:.1f} ms")

except KeyboardInterrupt:
    print("🛑 Stopped by user")
finally:
    cap.release()
