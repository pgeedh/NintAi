import cv2
import argparse
import sys
import os
import time
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import core
from src import tracking
from src import report
from src import ai_report

def main():
    parser = argparse.ArgumentParser(description="BikeFit Professional Video Analysis")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input video")
    parser.add_argument("--output_video", "-ov", type=str, help="Path to save the analyzed video")
    parser.add_argument("--output_excel", "-oe", type=str, default="output/bikefit_data.xlsx", help="Path to save the Excel data")
    parser.add_argument("--api_key", type=str, help="Google Gemini API Key")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_excel), exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open video at {args.input}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    print("Initializing YOLOv8 Tracking...")
    detector = tracking.PoseDetector()

    data = {'frame': [], 'time': [], 'knee': [], 'hip': [], 'elbow': []}
    
    frame_count = 0
    print(f"Processing video: {args.input}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # YOLO Prediction
        results = detector.predict(frame)
        
        # Default angles
        knee, hip, elbow = 0, 0, 0
        
        if results and results.keypoints and len(results.keypoints) > 0:
            lm_dict = detector.get_landmarks_dict(results, frame.shape)
            if lm_dict:
                angles = core.analyze_posture(lm_dict)
                knee = angles['knee']
                hip = angles['hip']
                elbow = angles['elbow']

                # Draw Visuals (Simple Skeleton)
                for k, v in lm_dict.items():
                    if v[0] != 0:
                        cv2.circle(frame, (int(v[0]), int(v[1])), 5, (0, 255, 255), -1)

        # Store Data
        data['frame'].append(frame_count)
        data['time'].append(frame_count / fps)
        data['knee'].append(knee)
        data['hip'].append(hip)
        data['elbow'].append(elbow)

        # Overlay Stats
        cv2.rectangle(frame, (0, 0), (250, 150), (0, 0, 0), -1) 
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Knee: {int(knee)}", (10, 30), font, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Hip: {int(hip)}", (10, 60), font, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Elbow: {int(elbow)}", (10, 90), font, 0.7, (0, 255, 255), 2)

        if out:
            out.write(frame)
        
        cv2.imshow('NintAi Video Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # --- Analysis & Report ---
    if len(data['knee']) > 0:
        df = pd.DataFrame(data)
        # Filter 0s
        df_clean = df[df['knee'] > 0]
        
        if not df_clean.empty:
            avg_knee = df_clean['knee'].mean()
            avg_elbow = df_clean['elbow'].mean()
            avg_hip = df_clean['hip'].mean()
            
            feedback_lines, saddle_adj, elbow_adj = core.get_feedback(avg_knee, avg_elbow)
            
            print("\n--- Fit Report ---")
            print(f"Avg Knee Angle: {avg_knee:.1f}°")
            print(f"Avg Elbow Angle: {avg_elbow:.1f}°")
            for line in feedback_lines:
                print(f"- {line}")
            
            df.to_excel(args.output_excel, index=False)
            print(f"Data saved to {args.output_excel}")

            # PDF Report
            report_path = args.output_excel.replace(os.path.splitext(args.output_excel)[1], ".pdf")
            recommendations = feedback_lines.copy()
            if saddle_adj['value'] > 0:
                recommendations.append(f"Saddle: {saddle_adj['direction']} {saddle_adj['value']:.1f} cm (Est)")
            if elbow_adj['value'] > 0:
                recommendations.append(f"Handlebar: {elbow_adj['direction']} {elbow_adj['value']:.1f} cm (Est)")
            
            print("Generating AI Analysis (Gemini)...")
            angles_summary = {'knee': avg_knee, 'elbow': avg_elbow, 'hip': avg_hip}
            ai_text = ai_report.generate_ai_analysis(angles_summary, recommendations, api_key=args.api_key)
            if args.api_key:
                print("  - Gemini API Key provided. Detailed report generated.")

            print(f"Generating PDF report to {report_path}...")
            report.generate_report("none", angles_summary, recommendations, report_path, ai_text)
        else:
            print("No valid pose data collected.")

if __name__ == "__main__":
    main()
