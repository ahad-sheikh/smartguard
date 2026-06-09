# SmartGuard - AI-Powered Automated Weapon Detection System

SmartGuard is an intelligent security surveillance application engineered to identify weapons (such as guns and knives) and detect behavioral anomalies in real-time. By pairing a custom Deep Learning computer vision pipeline with a responsive Django web backend, SmartGuard provides security personnel with a live camera feed monitor, dynamic alert logs, configuration settings, and an analytics dashboard.

## 🚀 Key Features
* **Real-Time Threat Detection:** Deep learning pipelines running object detection optimized for localized security feeds.
* **Granular Weapon Configuration:** Interactive settings dashboard allowing administrators to toggle tracking flags (`detect_gun`, `detect_knife`, etc.) on or off seamlessly.
* **Comprehensive Metrics & Logging:** Automated database logging to record, aggregate, and inspect daily incident counts and historical logs.
* **Modular Dashboard UI:** Dedicated client view templates for specialized operational tasks (`monitor`, `analytics`, `incidents`, `devices`, `settings`).

---

## 📂 Repository Architecture

```text
smartguard/
│
├── smartguard_project/     # Core Django configuration (settings, URLs, WSGI/ASGI)
├── detection/              # Core app containing application logic
│   ├── migrations/         # Database state migration logs
│   ├── models.py           # Database definitions (System settings, incident logs)
│   ├── views.py            # Backend dashboard logic and configuration handling
│   └── urls.py             # App-level routing and endpoints
│
├── templates/              # HTML Frontend Templates
│   └── dashboard/          # Extended UI layouts (monitor, analytics, settings, etc.)
│
├── detect.py               # Main execution script for the deep learning pipeline
├── requirements.txt        # Local environment Python package dependencies
└── .gitignore              # Files intentionally untracked by Git (db.sqlite3, weights)
🛠️ Local Installation & Execution Guide
Follow these steps to configure and boot up SmartGuard on your local machine:

1. Clone the Repository
Bash
git clone [https://github.com/ahad-sheikh/smartguard.git](https://github.com/ahad-sheikh/smartguard.git)
cd smartguard
2. Configure and Activate Environment
Ensure you have Anaconda or Miniconda installed on your host system. Spin up your local environment and download the compiled dependencies manifest:

Bash
conda activate smartguard
pip install -r requirements.txt
3. Supply the Deep Learning Weights
Because large model checkpoints are excluded from version control to prevent system repository bloat, you must manually provide the detection weights.

Download your target detection weights (e.g., yolo11n.pt).

Place the file directly in the project root directory (smartguard/).

### 📱 Using DroidCam as a Phone Camera
If you want to use your phone camera via DroidCam instead of an IP camera app:
1. Install the DroidCam app on your phone and the DroidCam client on your Windows PC.
2. Connect your phone via USB or Wi-Fi and start the DroidCam client.
3. Note the camera index shown by DroidCam (usually `0`, `1`, or `2`).
4. Open the SmartGuard dashboard, go to Settings, select `DroidCam (Windows client)` and enter that device index or stream URL.

4. Apply Database Migrations
Initialize the structural setup for your local database layer. This command will auto-generate your clean local db.sqlite3 instance:

Bash
python manage.py migrate
5. Launch the Application Server
Run the local deployment server:

Bash
python manage.py runserver
Once initialized, open your web browser and navigate to:
👉 http://127.0.0.1:8000/

🛡️ License
This project is developed for internal educational and research purposes as part of a final year project. All rights reserved.


# 1. Stage the markdown file
git add README.md

# 2. Commit it with a clear message
git commit -m "Add comprehensive project README documentation"

# 3. Push it live to GitHub
git push origin main