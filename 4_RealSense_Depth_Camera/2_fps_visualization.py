import cv2
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

try:
    while True:
        start_time = time.time()
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        results = model(frame, verbose=False, device='0')

        center_hx = center_hy = center_mx = center_my = 0
        area_h = area_m = 0

        for result in results:
            boxes = result.boxes
            if boxes is not None and boxes.xyxy is not None:
                xyxys = boxes.xyxy.cpu().numpy().round().astype(int)
                clss = boxes.cls.cpu().numpy().astype(int)

                for xyxy, cls_id in zip(xyxys, clss):
                    x1 = xyxy[0]
                    y1 = xyxy[1]
                    x2 = xyxy[2]
                    y2 = xyxy[3]

                    if cls_id == 0:
                        center_hx = (x1 + x2) // 2
                        center_hy = (y1 + y2) // 2
                        area_h = (x2 - x1) * (y2 - y1)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (center_hx, center_hy), 4, (0, 255, 0), -1)
                        cv2.putText(frame, "Object 1", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        print("Object 1:", area_h)

                    elif cls_id == 1:
                        center_mx = (x1 + x2) // 2
                        center_my = (y1 + y2) // 2
                        area_m = (x2 - x1) * (y2 - y1)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.circle(frame, (center_mx, center_my), 4, (0, 0, 255), -1)
                        cv2.putText(frame, "Object 2", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        print("Object 2:", area_m)

        if area_h != 0 and area_m != 0:
            mid_x = (center_hx + center_mx) // 2
            mid_y = (center_hy + center_my) // 2
            cv2.circle(frame, (mid_x, mid_y), 4, (255, 0, 0), -1)

        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("YOLO11n + Realsense Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
