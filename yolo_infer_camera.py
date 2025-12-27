import cv2
from ultralytics import YOLO

model = YOLO("runs/cube/cube_vs_notcube/weights/best.pt")
cap = cv2.VideoCapture(0)

CONF_THRESHOLD = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.3, iou=0.6, verbose=False)

    detected_cube = False

    for box in results[0].boxes:
        cls = int(box.cls)
        conf = float(box.conf)

        # ONLY show cube detections
        if cls == 0 and conf >= CONF_THRESHOLD:
            detected_cube = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Cube {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # If no cube detected → show NOT CUBE text (NO BOXES)
    if not detected_cube:
        cv2.putText(
            frame,
            "Not Cube",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Cube Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
