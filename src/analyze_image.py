import cv2
import argparse
import sys
import os
import numpy as np

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import core
from src import tracking
from src import report
from src import ai_report

def main():
    parser = argparse.ArgumentParser(description="BikeFit Professional Image Analysis")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output", "-o", type=str, help="Path to save the annotated image")
    parser.add_argument("--calibrate", action="store_true", help="Enable manual calibration mode (click points on wheel)")
    parser.add_argument("--api_key", type=str, help="Google Gemini API Key (or set GOOGLE_API_KEY env var)")
    args = parser.parse_args()

    # Load Image
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not load image at {args.input}")
        sys.exit(1)

    # --- Calibration Step ---
    calibrator = core.BikeCalibrator()
    if args.calibrate:
        print("--- CALIBRATION MODE ---")
        print("Please click the two endpoints of a known reference object (e.g., wheel diameter).")
        print("Press any key after selecting two points.")
        
        points = []
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Calibration", image)
                if len(points) == 2:
                    cv2.line(image, points[0], points[1], (0, 0, 255), 2)
                    cv2.imshow("Calibration", image)

        cv2.imshow("Calibration", image)
        cv2.setMouseCallback("Calibration", mouse_callback)
        cv2.waitKey(0)
        cv2.destroyWindow("Calibration")
        
        if len(points) == 2:
            ref_cm = 62.2 
            calibrator.calibrate(points[0], points[1], ref_cm) 
            print(f"Calibration successful: {calibrator.px_per_cm:.2f} px/cm")
        else:
            print("Calibration skipped (insufficient points).")
            
    # Initialize Tracking
    print("Loading Tracking Engine (YOLOv8)...")
    detector = tracking.PoseDetector() 

    # Process Image
    results = detector.predict(image)
    if not results:
        print("No pose detected.")
        sys.exit(1)

    lm_dict = detector.get_landmarks_dict(results, image.shape)
    if not lm_dict:
        print("Pose detected but landmarks mapping failed.")
        sys.exit(1)

    # Analyze Posture
    angles = core.analyze_posture(lm_dict)
    knee_angle = angles['knee']
    hip_angle = angles['hip']
    elbow_angle = angles['elbow']

    # Get Feedback
    feedback_lines, saddle_adj, elbow_adj = core.get_feedback(knee_angle, elbow_angle)

    # Draw Visuals (Landmarks & Skeleton)
    for k, v in lm_dict.items():
        if v[0] == 0 and v[1] == 0: continue
        cv2.circle(image, (int(v[0]), int(v[1])), 6, (0, 255, 255), -1)
        cv2.putText(image, k, (int(v[0])+10, int(v[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    skeleton = [('shoulder', 'elbow'), ('elbow', 'wrist'), ('shoulder', 'hip'), ('hip', 'knee'), ('knee', 'ankle'), ('ankle', 'foot')]
    for p1, p2 in skeleton:
        if p1 in lm_dict and p2 in lm_dict:
            pt1 = (int(lm_dict[p1][0]), int(lm_dict[p1][1]))
            pt2 = (int(lm_dict[p2][0]), int(lm_dict[p2][1]))
            if pt1 != (0,0) and pt2 != (0,0):
                cv2.line(image, pt1, pt2, (255, 255, 255), 2)

    # Overlay
    overlay = image.copy()
    panel_h, panel_w = 300, 450
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color_text = (255, 255, 255)
    color_val = (0, 255, 255)
    thick = 1
    line_spacing = 30
    x_start = 20
    y_start = 40

    cv2.putText(image, "NintAi Professional Fit", (x_start, y_start), font, 0.8, (0, 255, 0), 2)
    y_start += 10
    
    cv2.putText(image, "Knee Angle:", (x_start, y_start + line_spacing), font, font_scale, color_text, thick)
    cv2.putText(image, f"{int(knee_angle)} deg", (x_start + 120, y_start + line_spacing), font, font_scale, color_val, 2)
    
    cv2.putText(image, "Hip Angle:", (x_start, y_start + 2*line_spacing), font, font_scale, color_text, thick)
    cv2.putText(image, f"{int(hip_angle)} deg", (x_start + 120, y_start + 2*line_spacing), font, font_scale, color_val, 2)
    
    cv2.putText(image, "Elbow Angle:", (x_start, y_start + 3*line_spacing), font, font_scale, color_text, thick)
    cv2.putText(image, f"{int(elbow_angle)} deg", (x_start + 125, y_start + 3*line_spacing), font, font_scale, color_val, 2)

    y_feedback = y_start + 4 * line_spacing + 10
    cv2.putText(image, "Adjustments:", (x_start, y_feedback), font, 0.7, (0, 255, 0), 2)
    y_feedback += 30

    if saddle_adj['value'] > 0:
        cv2.putText(image, f"Saddle: {saddle_adj['direction']} {saddle_adj['value']:.1f} cm (Est)", 
                   (x_start, y_feedback), font, 0.6, (0, 100, 255), 2)
        y_feedback += 30

    if elbow_adj['value'] > 0:
        cv2.putText(image, f"Handlebar: {elbow_adj['direction']} {elbow_adj['value']:.1f} cm (Est)", 
                   (x_start, y_feedback), font, 0.6, (0, 100, 255), 2)

    # Save outputs
    if args.output:
        cv2.imwrite(args.output, image)
        print(f"Image result saved to {args.output}")

    # Generate PDF Report with AI
    full_recs = feedback_lines.copy()
    if saddle_adj['value'] > 0:
        full_recs.append(f"Saddle: {saddle_adj['direction']} {saddle_adj['value']:.1f} cm")
    if elbow_adj['value'] > 0:
        full_recs.append(f"Handlebar: {elbow_adj['direction']} {elbow_adj['value']:.1f} cm")

    print("Generating AI Analysis (Gemini)...")
    ai_text = ai_report.generate_ai_analysis(angles, full_recs, api_key=args.api_key)
    if args.api_key:
        print("  - Gemini API Key provided. Analysis should be detailed.")
    else:
        print("  - No API Key found. Analysis will be skipped or limited.")
    
    report_path = "output/report.pdf"
    if args.output:
        base, _ = os.path.splitext(args.output)
        report_path = f"{base}.pdf"
    
    print(f"Generating PDF report to {report_path}...")
    try:
        # BUG FIX: Use the annotated OUTPUT image (JPG) if available, to avoid WebP errors in PDF
        img_source = args.output if args.output else args.input
        report.generate_report(img_source, angles, full_recs, report_path, ai_text)
    except Exception as e:
        print(f"Warning: PDF generation failed. {e}")

    if not args.output:
        print("Displaying result window. Press any key to close.")
        cv2.imshow('NintAi Result', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
