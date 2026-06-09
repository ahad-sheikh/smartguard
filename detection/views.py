import cv2
import json
import time
import threading
from datetime import date, timedelta
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from .models import Incident, SystemSettings, SystemLog

# ── Global detection state ────────────────────────────────────
latest_detections = []
latest_stats = {'total': 0, 'guns': 0, 'knives': 0, 'armed': 0}
last_threat_time = 0
detection_lock = threading.Lock()

# ── Load YOLO models once ─────────────────────────────────────
weapon_model = None
person_model = None
use_openvino = False

def load_models():
    global weapon_model, person_model, use_openvino
    try:
        # Try OpenVINO first
        try:
            from openvino.runtime import Core
            import os
            ie = Core()
            
            weapon_xml = "D:/smartguard/models/best_openvino_model/best.xml"
            person_xml = "yolo11n_openvino_model/yolo11n.xml"
            
            if os.path.exists(weapon_xml) and os.path.exists(person_xml):
                weapon_model = ie.read_model(model=weapon_xml)
                person_model = ie.read_model(model=person_xml)
                weapon_model = ie.compile_model(weapon_model, device_name="CPU")
                person_model = ie.compile_model(person_model, device_name="CPU")
                use_openvino = True
                SystemLog.objects.create(
                    message="✓ OpenVINO models loaded successfully — CPU-optimized inference enabled (40-60% faster)",
                    level='info'
                )
            else:
                raise FileNotFoundError("OpenVINO .xml files not found")
        except Exception as ov_err:
            # Fallback to YOLO
            from ultralytics import YOLO
            weapon_model = YOLO("D:/smartguard/models/best.pt")
            person_model = YOLO("yolo11n.pt")
            use_openvino = False
            SystemLog.objects.create(
                message=f"YOLO models loaded (OpenVINO not available: {str(ov_err)}). Run 'yolo export model=best.pt format=openvino' to enable CPU optimization.",
                level='warn'
            )
    except Exception as e:
        SystemLog.objects.create(message=f"Critical model load error: {e}", level='error')

# Load models when server starts
threading.Thread(target=load_models, daemon=True).start()


def run_inference_yolo(model, frame, conf=0.5, classes=None):
    """Run YOLO inference (fallback)"""
    results = model.predict(source=frame, conf=conf, imgsz=320, verbose=False, classes=classes)
    return results[0].boxes if results[0].boxes is not None else None

def run_inference_openvino(model, frame, conf_threshold=0.5):
    """Run OpenVINO inference (optimized)"""
    import numpy as np
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (320, 240))
    blob = resized.transpose((2, 0, 1))[np.newaxis, :, :, :].astype(np.float32) / 255.0
    
    input_name = list(model.inputs)[0].get_any_name()
    results = model([blob])
    output = results[list(model.outputs)[0]]
    
    # Parse OpenVINO output (format may vary, fallback to simpler handling)
    return output if output is not None else None


# ── Video generator ───────────────────────────────────────────
def get_camera_source():
    s = SystemSettings.get()
    source = s.iphone_url.strip()
    if s.camera_source == 'webcam':
        return 0
    if s.camera_source == 'droidcam':
        return int(source) if source.isdigit() else source
    if s.camera_source == 'iphone':
        return source
    return source


def get_camera_label():
    s = SystemSettings.get()
    if s.camera_source == 'webcam':
        return 'Webcam'
    if s.camera_source == 'droidcam':
        return 'DroidCam'
    if s.camera_source == 'iphone':
        return 'IP Camera'
    return s.camera_source.title()


def open_camera(source):
    if isinstance(source, int):
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def get_center(box):
    x1, y1, x2, y2 = box.xyxy[0]
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def distance(c1, c2):
    return ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) ** 0.5

def box_width(box):
    x1, y1, x2, y2 = box.xyxy[0]
    return abs(x2 - x1)

def generate_frames():
    global latest_detections, latest_stats, last_threat_time
    weapon_classes = ['gun', 'knife', 'sword']
    frame_count = 0
    person_cache = []

    source = get_camera_source()
    cap = open_camera(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        current_source = get_camera_source()
        if current_source != source:
            cap.release()
            source = current_source
            cap = open_camera(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            cap.release()
            time.sleep(2)
            source = get_camera_source()
            cap = open_camera(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        ret, frame = cap.read()
        if not ret:
            cap.release()
            time.sleep(2)
            source = get_camera_source()
            cap = open_camera(source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        frame_count += 1
        settings = SystemSettings.get()
        conf_threshold = settings.confidence / 100
        skip = settings.frame_skip

        small = cv2.resize(frame, (320, 240))
        sx = frame.shape[1] / 320
        sy = frame.shape[0] / 240

        current_detections = []
        weapon_detections = []

        # Run weapon detection every frame
        if weapon_model:
            w_results = weapon_model.predict(source=small, conf=conf_threshold, imgsz=320, verbose=False)
            w_boxes = w_results[0].boxes
            if w_boxes is not None and len(w_boxes) > 0:
                for box in w_boxes:
                    cls_id = int(box.cls[0])
                    label  = weapon_classes[cls_id]
                    conf   = float(box.conf[0])

                    # Filter by settings
                    if label == 'gun' and not settings.detect_gun: continue
                    if label == 'knife' and not settings.detect_knife: continue
                    if label == 'sword' and not settings.detect_sword: continue

                    x1,y1,x2,y2 = box.xyxy[0]
                    x1,y1,x2,y2 = int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)
                    center = ((x1+x2)/2, (y1+y2)/2)
                    width  = abs(x2-x1)
                    weapon_detections.append((label, conf, center, width, (x1,y1,x2,y2)))

                    color = (0,0,255) if label=='gun' else (0,165,255) if label=='knife' else (0,255,255)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.0%}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Run person detection every N frames
        if frame_count % skip == 0 and person_model:
            p_results = person_model.predict(source=small, conf=0.5, classes=[0], imgsz=320, verbose=False)
            p_boxes = p_results[0].boxes
            person_cache = []
            if p_boxes is not None and len(p_boxes) > 0:
                for box in p_boxes:
                    x1,y1,x2,y2 = box.xyxy[0]
                    x1,y1,x2,y2 = int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy)
                    center = ((x1+x2)/2, (y1+y2)/2)
                    person_cache.append((center, (x1,y1,x2,y2)))
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)
                    cv2.putText(frame, "person", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

        # Armed person logic
        save_incident = False
        incident_data = {}
        for w_label, w_conf, w_center, w_width, w_coords in weapon_detections:
            near_person = False
            if settings.armed_person:
                for p_center, _ in person_cache:
                    if distance(w_center, p_center) < w_width * 3:
                        near_person = True
                        break
            if near_person:
                current_detections.append({'type':'armed','weapon':w_label,'conf':int(w_conf*100)})
                incident_data = {'weapon_type': w_label, 'detection_type': 'armed', 'confidence': round(w_conf*100, 1)}
                save_incident = True
                # Red alert border
                cv2.rectangle(frame, (0,0), (frame.shape[1],frame.shape[0]), (0,0,255), 6)
                cv2.putText(frame, f"ARMED PERSON: {w_label.upper()}", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            else:
                current_detections.append({'type':'weapon','weapon':w_label,'conf':int(w_conf*100)})
                if not save_incident:
                    incident_data = {'weapon_type': w_label, 'detection_type': 'weapon', 'confidence': round(w_conf*100, 1)}
                    save_incident = True

        # Save incident to DB (throttled — max 1 per 5 seconds)
        now = time.time()
        if save_incident and (now - last_threat_time) > 5:
            last_threat_time = now
            try:
                Incident.objects.create(**incident_data)
            except Exception:
                pass

        # Update global state
        with detection_lock:
            latest_detections = current_detections
            total = Incident.objects.count()
            guns  = Incident.objects.filter(weapon_type='gun').count()
            knives= Incident.objects.filter(weapon_type='knife').count()
            armed = Incident.objects.filter(detection_type='armed').count()
            latest_stats = {'total': total, 'guns': guns, 'knives': knives, 'armed': armed}

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ── Views ─────────────────────────────────────────────────────

def video_feed(request):
    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def latest_detection_api(request):
    with detection_lock:
        dets = list(latest_detections)
        stats = dict(latest_stats)

    global last_threat_time
    new_threat = len(dets) > 0 and (time.time() - last_threat_time) < 3

    return JsonResponse({'detections': dets, 'stats': stats, 'new_threat': new_threat})


def dashboard(request):
    recent_alerts = Incident.objects.all()[:5]
    system_logs   = SystemLog.objects.all()[:6]
    total_threats = Incident.objects.count()
    gun_count     = Incident.objects.filter(weapon_type='gun').count()
    knife_count   = Incident.objects.filter(weapon_type='knife').count()
    armed_count   = Incident.objects.filter(detection_type='armed').count()
    alert_count   = Incident.objects.filter(timestamp__date=date.today()).count()
    feed_source   = get_camera_label()

    return render(request, 'dashboard/monitor.html', {
        'recent_alerts': recent_alerts,
        'system_logs':   system_logs,
        'total_threats': total_threats,
        'gun_count':     gun_count,
        'knife_count':   knife_count,
        'armed_count':   armed_count,
        'alert_count':   alert_count,
        'feed_source_name': feed_source,
    })


def incidents(request):
    incidents_qs = Incident.objects.all()
    today = date.today()
    return render(request, 'dashboard/incidents.html', {
        'incidents':   incidents_qs,
        'total_count': incidents_qs.count(),
        'gun_count':   incidents_qs.filter(weapon_type='gun').count(),
        'knife_count': incidents_qs.filter(weapon_type='knife').count(),
        'sword_count': incidents_qs.filter(weapon_type='sword').count(),
        'armed_count': incidents_qs.filter(detection_type='armed').count(),
        'today_count': incidents_qs.filter(timestamp__date=today).count(),
        'alert_count': incidents_qs.filter(timestamp__date=today).count(),
    })


def clear_incidents(request):
    Incident.objects.all().delete()
    return redirect('incidents')


def analytics(request):
    incidents_qs = Incident.objects.all()
    total = incidents_qs.count()
    gun_c   = incidents_qs.filter(weapon_type='gun').count()
    knife_c = incidents_qs.filter(weapon_type='knife').count()
    sword_c = incidents_qs.filter(weapon_type='sword').count()
    armed_c = incidents_qs.filter(detection_type='armed').count()

    # Most common
    most_common = 'gun'
    if knife_c > gun_c and knife_c > sword_c: most_common = 'knife'
    elif sword_c > gun_c and sword_c > knife_c: most_common = 'sword'

    # Last 7 days chart
    today = date.today()
    labels = []
    data   = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        labels.append(d.strftime('%b %d'))
        data.append(incidents_qs.filter(timestamp__date=d).count())

    today_count = incidents_qs.filter(timestamp__date=today).count()

    return render(request, 'dashboard/analytics.html', {
        'total_count': total,
        'gun_count':   gun_c,
        'knife_count': knife_c,
        'sword_count': sword_c,
        'armed_count': armed_c,
        'most_common': most_common,
        'chart_labels': json.dumps(labels),
        'chart_data':   json.dumps(data),
        'alert_count':  today_count,
    })


def settings_view(request):
    s = SystemSettings.get()
    saved = False
    if request.method == 'POST':
        s.camera_source = request.POST.get('camera_source', 'webcam')
        s.iphone_url    = request.POST.get('iphone_url', '').strip()
        s.resolution    = request.POST.get('resolution', '320x240')
        s.confidence    = float(request.POST.get('confidence', 50))
        s.frame_skip    = int(request.POST.get('frame_skip', 3))
        s.detect_gun    = 'detect_gun' in request.POST
        s.detect_knife  = 'detect_knife' in request.POST
        s.detect_sword  = 'detect_sword' in request.POST
        s.armed_person  = 'armed_person' in request.POST
        s.save()
        saved = True
        SystemLog.objects.create(message="Settings updated by user", level='info')

    today_count = Incident.objects.filter(timestamp__date=date.today()).count()
    return render(request, 'dashboard/settings.html', {'settings': s, 'saved': saved, 'alert_count': today_count})


def devices(request):
    s = SystemSettings.get()
    device_list = []
    if s.camera_source == 'iphone' and s.iphone_url:
        device_list.append({'name': 'IP Camera Stream', 'url': s.iphone_url, 'status': 'online', 'type': 'IP CAMERA STREAM'})
    elif s.camera_source == 'droidcam' and s.iphone_url:
        device_list.append({'name': 'DroidCam', 'url': s.iphone_url, 'status': 'online', 'type': 'DROIDCAM'})
    today_count = Incident.objects.filter(timestamp__date=date.today()).count()
    return render(request, 'dashboard/devices.html', {'devices': device_list, 'alert_count': today_count})
