# NintAi: The Ultimate Bike Fit Studio

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-SOTA%20Pose-purple)
![Gemini 3](https://img.shields.io/badge/AI-Google%20Gemini%203-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

**NintAi** is a professional-grade bio-mechanical analysis tool that turns your computer into a high-end bike fitting studio. Inspired by industry leaders like *MyVeloFit*, NintAi uses **YOLOv8** for tracking and **Google Gemini 3 (AI)** to provide expert-level written reports.

---

## 🚀 Key Features

*   **⚡ SOTA Tracking**: Ultralytics YOLOv8 for robust pose estimation.
*   **🤖 AI Coach (Gemini)**: Uses Google's LLM to generate a detailed, human-like analysis of your fit in the PDF report.
*   **📏 Precise Calibration**: Interactive "Wheel Calibration" for cm-perfect adjustments.
*   **📄 PDF Reports**: Sleek summaries with your photo, data, and AI insights.

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

### 1. Image Analysis (Studio Mode)
```bash
python src/analyze_image.py --input assets/examples/images/3.webp --calibrate --api_key "YOUR_KEY"
```
*   **`--calibrate`**: Click 2 points on your wheel to define scale.
*   **`--api_key`**: Enables the AI analysis in the PDF report.
*   **Output**: A window showing the fit + `output/report.pdf` with AI feedback.

### 2. Video Analysis (Motion Mode)
```bash
python src/analyze_video.py --input assets/examples/videos/testvideo2.mp4 --api_key "YOUR_KEY"
```
*   Tracks your motion over time and gives an average fit report with AI insights.

---

## 📁 Project Structure

```
NintAi/
├── src/
│   ├── analyze_image.py # Main Image Script
│   ├── analyze_video.py # Main Video Script
│   ├── ai_report.py    # Gemini Integration
│   ├── core.py         # Biomechanics Logic
│   ├── tracking.py     # YOLOv8
│   └── report.py       # PDF Generator
```

---

##  License

MIT License.