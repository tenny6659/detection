from ultralytics import YOLO
import cv2

model = YOLO("runs/cube/cube_force/weights/best.pt")


img = cv2.imread("cube_dataset/images/train/0.jpg")

results = model.predict(
    img,
    conf=0.001,     # 🔥 EXTREMELY LOW
    iou=0.1,
    verbose=True
)

annotated = results[0].plot(
    labels=True,
    conf=True
)

cv2.imshow("Test", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Class names:", model.names)

