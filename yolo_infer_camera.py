import cv2
from ultralytics import YOLO

model = YOLO("runs/cube/cube_camera/weights/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        frame,
        conf=0.35,
        iou=0.5,
        verbose=False
    )

    annotated = results[0].plot()
    cv2.imshow("Cube Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
