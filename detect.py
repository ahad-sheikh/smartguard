from ultralytics import YOLO
import cv2
import time

# ── Load Models ───────────────────────────────────────────────
weapon_model = YOLO("D:/smartguard/models/best.pt")
person_model  = YOLO("yolo11n.pt")

weapon_classes = ['gun', 'knife', 'sword']


SOURCE = "http://192.168.0.102:8081/video"
# ── Helper Functions ──────────────────────────────────────────
def get_center(box):
    x1, y1, x2, y2 = box.xyxy[0]
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def distance(c1, c2):
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5

def box_width(box):
    x1, y1, x2, y2 = box.xyxy[0]
    return abs(x2 - x1)

def connect_camera(source):
    print(f"Connecting to: {source}")
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        print("Connected successfully!")
    else:
        print("Connection failed — retrying...")
    return cap

# ── Main Loop with Auto-Reconnect ─────────────────────────────
frame_count = 0
person_detections_cache = []

print("SmartGuard started. Press Q to quit.")

while True:
    cap = connect_camera(SOURCE)

    if not cap.isOpened():
        print("Waiting 3 seconds before retry...")
        time.sleep(3)
        continue

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Stream lost — reconnecting in 3 seconds...")
            cap.release()
            time.sleep(3)
            break  # go back to outer loop to reconnect

        frame_count += 1

        # ── Resize for faster inference ───────────────────────
        small_frame = cv2.resize(frame, (320, 240))

        # ── Weapon detection every frame ──────────────────────
        weapon_results = weapon_model.predict(
            source=small_frame,
            conf=0.5,
            imgsz=320,
            verbose=False
        )

        # ── Person detection every 3rd frame ──────────────────
        if frame_count % 3 == 0:
            person_results = person_model.predict(
                source=small_frame,
                conf=0.5,
                classes=[0],
                imgsz=320,
                verbose=False
            )
            person_boxes = person_results[0].boxes
            person_detections_cache = []
            if person_boxes is not None and len(person_boxes) > 0:
                for box in person_boxes:
                    person_detections_cache.append((get_center(box), box))

        display_frame = frame.copy()
        sx = frame.shape[1] / 320
        sy = frame.shape[0] / 240

        weapon_detections = []

        # ── Draw weapon boxes ─────────────────────────────────
        weapon_boxes = weapon_results[0].boxes
        if weapon_boxes is not None and len(weapon_boxes) > 0:
            for box in weapon_boxes:
                cls_id = int(box.cls[0])
                label  = weapon_classes[cls_id]
                conf   = float(box.conf[0])
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)
                center = ((x1+x2)/2, (y1+y2)/2)
                width  = abs(x2-x1)
                weapon_detections.append((label, conf, center, width))
                color = (0,0,255) if label=='gun' else \
                        (0,165,255) if label=='knife' else \
                        (0,255,255)
                cv2.rectangle(display_frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(display_frame, f"{label} {conf:.0%}",
                    (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ── Draw person boxes ─────────────────────────────────
        for p_center, p_box in person_detections_cache:
            x1,y1,x2,y2 = p_box.xyxy[0]
            x1,y1,x2,y2 = int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)
            cv2.rectangle(display_frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(display_frame, "person",
                (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ── Person with weapon logic ───────────────────────────
        alert_messages = []
        armed_detected = False

        for w_label, w_conf, w_center, w_width in weapon_detections:
            near_person = False
            for p_center, _ in person_detections_cache:
                pc = (p_center[0]*sx, p_center[1]*sy)
                if distance(w_center, pc) < w_width * 3:
                    near_person = True
                    break
            if near_person:
                armed_detected = True
                alert_messages.append(f"ARMED PERSON: {w_label} ({w_conf:.0%})")
            else:
                alert_messages.append(f"WEAPON DETECTED: {w_label} ({w_conf:.0%})")

        # ── Alert UI ──────────────────────────────────────────
        if alert_messages:
            border = (0,0,255) if armed_detected else (0,100,255)
            cv2.rectangle(display_frame, (0,0),
                (display_frame.shape[1], display_frame.shape[0]), border, 8)
            for i, msg in enumerate(alert_messages):
                cv2.putText(display_frame, msg,
                    (20, 50+i*40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,0,255), 2)
                print(msg)
        else:
            cv2.putText(display_frame, "No Threat Detected",
                (20,50), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,255,0), 2)

        cv2.imshow("SmartGuard - Weapon Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()