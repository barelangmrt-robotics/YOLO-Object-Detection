# =====================================
# 1. IMPORT LIBRARIES
# -------------------------------------
# Import necessary libraries:
# - ultralytics.YOLO for object detection
# - cv2 for camera input and display
# - time for FPS calculation
# - numpy for array operations
# =====================================
from ultralytics import YOLO
import cv2
import time
import numpy as np

# =====================================
# 2. CAMERA INITIALIZATION
# -------------------------------------
# Open the camera stream with given resolution.
# Returns None if the camera cannot be opened.
# =====================================
def init_camera(width=640, height=320, index=0):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("Failed to open camera")
        return None
    return cap

# =====================================
# 3. FRAME PRE-PROCESSING
# -------------------------------------
# Resize the frame and calculate scale factors
# to map detection results back to original frame size.
# =====================================
def preprocess_frame(frame, target_width=640, target_height=320):
    resized = cv2.resize(frame, (target_width, target_height))
    scale_x = frame.shape[1] / resized.shape[1]
    scale_y = frame.shape[0] / resized.shape[0]
    return resized, scale_x, scale_y

# =====================================
# 4. OBJECT DETECTION USING YOLO
# -------------------------------------
# Run YOLO model on the resized frame to detect objects.
# Returns detection results (bounding boxes, class IDs, etc.).
# =====================================
def predict_objects(model, frame_resized):
    return model(frame_resized, stream=True, verbose=False, device='0')

# =====================================
# 5. DRAW DETECTIONS & MIDPOINT
# -------------------------------------
# Draw bounding boxes and class labels on the original frame.
# Draw the midpoint if two specific objects are found.
# Class 0 = Object 1, Class 1 = Object 2
# =====================================
def draw_detections(frame, results, scale_x, scale_y):
    center_hx = center_hy = center_mx = center_my = 0
    area_h = area_m = 0

    for result in results:
        boxes = result.boxes
        if boxes is not None and boxes.xyxy is not None:
            xyxys = boxes.xyxy.cpu().numpy().round().astype(int)
            clss = boxes.cls.cpu().numpy().astype(int)

            for xyxy, cls_id in zip(xyxys, clss):
                x1 = int(xyxy[0] * scale_x)
                y1 = int(xyxy[1] * scale_y)
                x2 = int(xyxy[2] * scale_x)
                y2 = int(xyxy[3] * scale_y)

                # Object 1 (Class 0)
                if cls_id == 0:
                    center_hx = (x1 + x2) // 2
                    center_hy = (y1 + y2) // 2
                    area_h = (x2 - x1) * (y2 - y1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (center_hx, center_hy), 4, (0, 255, 0), -1)
                    cv2.putText(frame, "Object 1", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    print("Object 1:", area_h)

                # Object 2 (Class 1)
                elif cls_id == 1:
                    center_mx = (x1 + x2) // 2
                    center_my = (y1 + y2) // 2
                    area_m = (x2 - x1) * (y2 - y1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(frame, (center_mx, center_my), 4, (0, 0, 255), -1)
                    cv2.putText(frame, "Object 2", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    print("Object 2:", area_m)

    # Midpoint between Object 1 and Object 2
    if area_h != 0 and area_m != 0:
        mid_x = (center_hx + center_mx) // 2
        mid_y = (center_hy + center_my) // 2
        cv2.circle(frame, (mid_x, mid_y), 4, (255, 0, 0), -1)

    return frame

# =====================================
# 6. FPS CALCULATION
# -------------------------------------
# Calculate FPS from elapsed time.
# =====================================
def calculate_fps(start_time):
    elapsed = time.time() - start_time
    return 1 / elapsed if elapsed > 0 else 0

# =====================================
# 7. FPS STATISTICS
# -------------------------------------
# Display max, min, and average FPS every interval.
# =====================================
def print_fps_statistics(fps_list, interval):
    if fps_list:
        max_fps = max(fps_list)
        min_fps = min(fps_list)
        avg_fps = sum(fps_list) / len(fps_list)
        print(f"\n[FPS STATISTICS - {interval} seconds]")
        print(f"Max FPS: {max_fps:.2f}")
        print(f"Min FPS: {min_fps:.2f}")
        print(f"Average FPS: {avg_fps:.2f}\n")
        fps_list.clear()

# =====================================
# 8. MAIN VIDEO PROCESSING LOOP
# -------------------------------------
# Main loop that:
# - Captures camera frames
# - Preprocesses and runs detection
# - Visualizes detections
# - Calculates and displays FPS
# =====================================
def process_video(cap, model):
    fps_list = []
    interval = 5  # seconds
    start_time_global = time.time()

    while True:
        start_time = time.time()

        # --- 1. Capture frame from camera ---
        ret, frame = cap.read()
        if not ret:
            break

        # --- 2. Preprocess frame ---
        frame_resized, scale_x, scale_y = preprocess_frame(frame)

        # --- 3. Detect objects using YOLO ---
        results = predict_objects(model, frame_resized)

        # --- 4. Draw bounding boxes and centers ---
        frame = draw_detections(frame, results, scale_x, scale_y)

        # --- 5. Calculate FPS ---
        fps = calculate_fps(start_time)
        if 1 < fps < 150:
            fps_list.append(fps)

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # --- 6. Print FPS stats every interval ---
        if time.time() - start_time_global >= interval:
            print_fps_statistics(fps_list, interval)
            start_time_global = time.time()

        # --- 7. Show result window ---
        cv2.imshow("YOLO11n Object Detection", frame)

        # --- 8. Quit with 'q' key ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =====================================
# 9. MAIN EXECUTION BLOCK
# -------------------------------------
# - Load model
# - Initialize camera
# - Run detection process
# =====================================
if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    camera = init_camera()
    if camera:
        process_video(camera, model)
