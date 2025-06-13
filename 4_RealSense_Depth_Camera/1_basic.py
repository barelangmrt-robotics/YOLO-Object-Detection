import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    results = model(color_image, verbose=False, device=0)
    frame_with_detections = results[0].plot()

    cv2.imshow("YOLO11n + Realsense Object Detection", frame_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipeline.stop()
cv2.destroyAllWindows()
