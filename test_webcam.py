from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("D:/smartguard/models/best.pt")

# Class names
class_names = ['gun', 'knife', 'sword']

# Alert colors per class
colors = {
    'gun':   (0, 0, 255),    # Red
    'knife': (0, 165, 255),  # Orange
    'sword': (0, 255, 255),  # Yellow
}

# Open webcam
cap = cv2.VideoCapture(0)

print("Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model.predict(
        source=frame,
        conf=0.5,
        verbose=False
    )

    annotated_frame = results[0].plot()
    boxes = results[0].boxes

    alert_triggered = False
    detected_classes = []

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            label  = class_names[cls_id]
            conf   = float(box.conf[0])
            detected_classes.append(f"{label} ({conf:.0%})")
            alert_triggered = True

    # Draw alert border if weapon detected
    if alert_triggered:
        cv2.rectangle(annotated_frame, (0, 0),
            (annotated_frame.shape[1], annotated_frame.shape[0]),
            (0, 0, 255), 8)
        alert_text = "ALERT: " + ", ".join(detected_classes)
        cv2.putText(annotated_frame, alert_text,
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 0, 255), 3)
        print(alert_text)
    else:
        cv2.putText(annotated_frame, "No Threat Detected",
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0), 2)

    cv2.imshow("SmartGuard - Weapon Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()