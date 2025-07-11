import cv2
import numpy as np
import onnxruntime as ort
import pyvirtualcam
import time

# ------------------------
FRAME_WIDTH = 160     # Lower = faster (try 160x120, 320x240, etc.)
FRAME_HEIGHT = 120
FPS = 30
MODEL_PATH = "student_model.onnx"
# ------------------------

# Load ONNX model
print("🔄 Loading ONNX model...")
ort_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
print("✅ ONNX model loaded.")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Start virtual camera (expects BGR format)
with pyvirtualcam.Camera(width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=FPS, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
    print("🟢 Virtual camera started. Use 'PythonCam' or 'OBS Virtual Camera' in Zoom/Meet.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam.")
            break

        # Start timer
        start_time = time.time()

        # Convert BGR to RGB & normalize
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :]  # (1, 3, H, W)

        # ONNX inference
        ort_inputs = {"input": input_tensor.astype(np.float32)}
        ort_outs = ort_session.run(None, ort_inputs)
        output_tensor = ort_outs[0][0]  # (3, H, W)

        # Convert back to uint8 BGR image
        output_np = np.transpose(output_tensor, (1, 2, 0))  # (H, W, 3), RGB
        output_np = np.clip(output_np, 0, 1)
        output_np = np.power(output_np, 1 / 2.2)  # Optional gamma correction
        output_rgb = (output_np * 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

        # End timer & calculate FPS
        end_time = time.time()
        inference_time = end_time - start_time
        fps = 1.0 / inference_time
        print(f"Inference time: {inference_time * 1000:.2f} ms | FPS: {fps:.2f}")

        # Send to virtual cam
        cam.send(output_bgr)
        cam.sleep_until_next_frame()

cap.release()
cv2.destroyAllWindows()
