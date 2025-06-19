import time
import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect

# ==== CONFIG ====
MODEL_PATH = 'face_edgetpu.tflite'
SCORE_THRESHOLD = 0.6
FLIP_VERTICAL = True  # Set to True if camera image is vertically flipped
CAMERA_INDEX = 0
# ================

# Load model
print(f"🎯 Loading model: {MODEL_PATH}")
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()
input_size = common.input_size(interpreter)
print(f"✅ Model loaded. Input size: {input_size}")

# Start webcam
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("❌ Failed to open webcam")

print("🚀 Running Coral TPU face detection...")
frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Skipped frame (read failure)")
            continue

        # Flip if needed
        if FLIP_VERTICAL:
            frame = cv2.flip(frame, 0)

        # Resize to model input size
        resized_frame = cv2.resize(frame, input_size)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Run inference
        common.set_input(interpreter, rgb_frame)
        inference_start = time.time()
        interpreter.invoke()
        inference_time = (time.time() - inference_start) * 1000  # ms

        # Parse results
        objs = detect.get_objects(interpreter, score_threshold=SCORE_THRESHOLD)
        print(f"\n📷 Frame shape: {frame.shape}")
        print(f"🧠 {len(objs)} face(s) detected")
        for i, obj in enumerate(objs):
            print(f"  • Face {i+1}: Score={obj.score:.2f}, BBox={obj.bbox}")

        print(f"⏱️ Inference time: {inference_time:.1f} ms")

        # FPS tracking
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"📊 Avg FPS: {frame_count / elapsed:.2f}")

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user")

finally:
    cap.release()
    print("📷 Camera released")
