# NintAi: The Ultimate Bike Fit Studio

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Heavy%20Pose-orange)
![Gemini 3](https://img.shields.io/badge/AI-Google%20Gemini%203-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

**NintAi** is a professional-grade bio-mechanical analysis tool that turns your computer into a high-end bike fitting studio. Providing "Pro" level analysis comparable to industry leaders involving **MediaPipe Pose (Heavy)** for high-fidelity tracking (including heels/toes) and **Google Gemini 3 (AI)** for expert-level generated reports.

---

## 🚀 Key Features

*   **⚡ Pro-Grade Tracking**: Google's **MediaPipe Pose (Heavy)** engine provides smooth, jitter-free tracking of 33 keypoints including **Heels and Toes**.
*   **🦶 True Geometry**: Visualizes "Real Ankling" using actual foot geometry (no virtual estimation).
*   **🤖 AI Coach (Gemini)**: Uses Google's LLM to generate a detailed, human-like analysis of your fit in the PDF report.
*   **📊 Quad-View Reports**: Generates a professional 4-phase summary (Top, Front, Bottom, Overall) with "Red Dot" markers and metric overlays.
*   **📏 Precise Calibration**: Interactive "Wheel Calibration" for cm-perfect adjustments.
*   **📐 Bike Geometry**: Measure Frame Stack, Reach, and Saddle Height directly from the image.

---

## 📦 Installation

1.  **Clone the Repo**
    ```bash
    git clone https://github.com/yourusername/NintAi.git
    cd NintAi
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up AI (Optional)**
    Get a generic **Gemini API Key** from [Google AI Studio](https://aistudio.google.com/).
    You can pass it via CLI or set an environment variable:
    ```bash
    export GOOGLE_API_KEY="your_key_here"
    ```

---

## 🛠️ Usage

### 1. Video Analysis (The Ultimate Fit)
```bash
python src/analyze_video.py --input assets/examples/videos/testvideo2.mp4 --api_key "YOUR_KEY"
```
*   **Engine**: MediaPipe Pose (Tasks API).
*   **Output**: Generates `output/final_nintai_report.pdf` with the Quad-View layout and AI analysis.

### 2. Image Analysis (Quick Check)
```bash
python src/analyze_image.py --input assets/examples/images/3.webp --calibrate --measure_bike --api_key "YOUR_KEY"
```
*   **`--calibrate`**: Click 2 points on your wheel to define scale.
*   **`--measure_bike`**: Click BB, Saddle, and Handlebar to get Reach, Stack, and Saddle Height in cm.

---

## 📁 Project Structure

```
NintAi/
├── src/
│   ├── analyze_video.py # Main Video Script (MediaPipe)
│   ├── analyze_image.py # Main Image Script
│   ├── ai_report.py    # Gemini Integration
│   ├── core.py         # Biomechanics Logic
│   ├── tracking_mp.py  # MediaPipe Wrapper
│   ├── tracking.py     # Legacy YOLO Wrapper
│   ├── report.py       # PDF Generator
│   └── models/         # Downloaded Model Weights
```

---

## 📄 License

MIT License.
