import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
camera = cv2.VideoCapture(0)

while True:
    success, video_frame = camera.read()
    if not success:
        break

    detection_results = model(video_frame, verbose=False, device=0)
    frame_with_detections = detection_results[0].plot()

    cv2.imshow("YOLO11n Object Detection", frame_with_detections)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
