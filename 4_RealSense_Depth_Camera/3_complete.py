# =====================================
# 1. IMPORT LIBRARIES
# -------------------------------------
# Importing YOLO, OpenCV, time, NumPy, and pyrealsense2 for depth camera handling
# =====================================
from ultralytics import YOLO
import cv2
import time
import numpy as np
import pyrealsense2 as rs

# =====================================
# 2. INITIALIZE REALSENSE CAMERA
# -------------------------------------
# Set up RealSense RGB and Depth streams
# =====================================
def init_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)

    # Convert depth values to real-world distance (in meters)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align, depth_scale

# =====================================
# 3. FRAME PRE-PROCESSING
# -------------------------------------
# Resize the frame and calculate scale factors for mapping detections
# =====================================
def preprocess_frame(frame, target_width=480, target_height=320):
    resized = cv2.resize(frame, (target_width, target_height))
    scale_x = frame.shape[1] / resized.shape[1]
    scale_y = frame.shape[0] / resized.shape[0]
    return resized, scale_x, scale_y

# =====================================
# 4. YOLO OBJECT DETECTION
# -------------------------------------
# Perform object prediction using YOLO model
# =====================================
def predict_objects(model, frame_resized):
    return model(frame_resized, stream=True, verbose=False, device='0')

# =====================================
# 5. DRAW DETECTIONS & MIDPOINT WITH DEPTH
# -------------------------------------
# Draw bounding boxes, class labels, and midpoint on the original frame.
# Also calculate and display distance using depth data.
# Class 0 = Object 1, Class 1 = Object 2
# =====================================
def draw_detections_with_depth(frame, depth_frame, results, scale_x, scale_y, depth_scale):
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

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Get depth at center of bounding box
                depth_value = depth_frame.get_distance(cx, cy)
                distance_cm = depth_value * 100  # convert to cm

                if cls_id == 0:
                    center_hx, center_hy = cx, cy
                    area_h = (x2 - x1) * (y2 - y1)
                    color = (0, 255, 0)  # green
                    label = "Object 1"
                elif cls_id == 1:
                    center_mx, center_my = cx, cy
                    area_m = (x2 - x1) * (y2 - y1)
                    color = (0, 0, 255)  # red
                    label = "Object 2"
                else:
                    continue  # skip unknown classes

                # Draw bounding box and center circle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)

                # Put label with distance
                cv2.putText(frame, f"{label}: {distance_cm:.1f} cm", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                print(f"{label} area: {area_h if cls_id == 0 else area_m}, distance: {distance_cm:.1f} cm")

    # Draw midpoint if both objects detected
    if area_h != 0 and area_m != 0:
        mid_x = (center_hx + center_mx) // 2
        mid_y = (center_hy + center_my) // 2
        cv2.circle(frame, (mid_x, mid_y), 4, (255, 0, 0), -1)  # blue midpoint

    return frame

# =====================================
# 6. FPS CALCULATION & STATISTICS
# -------------------------------------
# Calculate FPS and display statistics periodically
# =====================================
def calculate_fps(start_time):
    elapsed = time.time() - start_time
    return 1 / elapsed if elapsed > 0 else 0

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
# 7. MAIN DETECTION LOOP (YOLO + REALSENSE)
# -------------------------------------
# Main loop to run object detection using RealSense input
# =====================================
def process_realsense(pipeline, align, model, depth_scale):
    fps_list = []
    interval = 5  # seconds
    start_time_global = time.time()

    while True:
        start_time = time.time()

        # Capture frames from RealSense
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert to NumPy array
        color_image = np.asanyarray(color_frame.get_data())

        # Pre-processing
        frame_resized, scale_x, scale_y = preprocess_frame(color_image)

        # Run object detection
        results = predict_objects(model, frame_resized)

        # Visualize detections and calculate distances
        color_image = draw_detections_with_depth(color_image, depth_frame, results, scale_x, scale_y, depth_scale)

        # Calculate FPS
        fps = calculate_fps(start_time)
        if 1 < fps < 150:
            fps_list.append(fps)

        # Overlay FPS on the frame
        cv2.putText(color_image, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Print FPS stats every interval
        if time.time() - start_time_global >= interval:
            print_fps_statistics(fps_list, interval)
            start_time_global = time.time()

        # Show result on screen
        cv2.imshow("YOLO11n + Realsense Object Detection", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()
    cv2.destroyAllWindows()

# =====================================
# 8. MAIN FUNCTION (MODEL & REALSENSE CALL)
# -------------------------------------
# Load YOLO model and start the RealSense detection pipeline
# =====================================
if __name__ == "__main__":
    # Load YOLO model
    model = YOLO("yolo11n.pt")

    # Initialize RealSense
    pipeline, align, depth_scale = init_realsense()

    # Start main detection loop
    process_realsense(pipeline, align, model, depth_scale)
