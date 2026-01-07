import cv2
import argparse
import sys
import os
import time
import pandas as pd
import numpy as np
import math

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import core
# Switch to MediaPipe
from src import tracking_mp as tracking 
from src import report
from src import ai_report

def main():
    parser = argparse.ArgumentParser(description="NintAi Ultimate BikeFit Tool (MediaPipe)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input video")
    parser.add_argument("--output_video", "-ov", type=str, help="Output video")
    parser.add_argument("--output_excel", "-oe", type=str, default="output/ultimate_data.xlsx", help="Output Excel")
    parser.add_argument("--api_key", type=str, help="Gemini API Key")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_excel), exist_ok=True)
    report_dir = os.path.dirname(args.output_excel)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened(): sys.exit(1)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = None
    if args.output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    print("Initializing NintAi Tracking (MediaPipe Tasks)...")
    # MP: model_complexity=2 (Heavy) for best accuracy
    # Model path is critical now
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/pose_landmarker_heavy.task'))
    if not os.path.exists(model_path):
        # Fallback to relative
        model_path = 'src/models/pose_landmarker_heavy.task'
    
    detector = tracking.PoseDetectorMP(model_path=model_path, model_complexity=2)

    # Filters
    filter_keys = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 
                   'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                   'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 
                   'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                   'left_heel', 'right_heel', 'left_toe', 'right_toe']
                   
    filters = {k: core.OneEuroFilter(t0=0, x0=np.zeros(2)) for k in filter_keys}

    frames_data = []
    
    side_votes = {'left': 0, 'right': 0}
    locked_side = None
    FRAMES_TO_LOCK = 30
    frame_count = 0

    print(f"Processing... {args.input}")
    t_start = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        t_curr = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t_start == 0: t_start = t_curr
        frame_count += 1
        
        # MP Tasks needs integer timestamp in ms
        ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        # Ensure strict monotonicity just in case
        if frame_count > 1 and ts_ms <= 0: ts_ms = frame_count * 33 # Approximate if meta missing

        results = detector.predict(frame, timestamp_ms=ts_ms)
        raw_lm = detector.get_landmarks_dict(results, frame.shape)
        
        clean_lm = {}
        for k, v in raw_lm.items():
            if k in filters: clean_lm[k] = filters[k](t_curr, np.array(v))
            else: clean_lm[k] = v
            
        if clean_lm:
            detected = core.detect_side(clean_lm)
            
            if locked_side is None:
                side_votes[detected] += 1
                if frame_count >= FRAMES_TO_LOCK:
                    locked_side = 'left' if side_votes['left'] >= side_votes['right'] else 'right'
                    print(f"Side Locked: {locked_side.upper()}")
                current_side = detected 
            else:
                current_side = locked_side
                
            unified_lm = core.get_primary_landmarks(clean_lm, current_side)
            angles = core.analyze_posture(unified_lm)
            
            frames_data.append({
                'frame_idx': int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                'angles': angles,
                'landmarks': unified_lm,
                'clean_lm': clean_lm, 
                'side': current_side
            })
            
            # --- Visuals ---
            
            # 1. Real Foot Visualization (MP has Heel/Toe)
            if 'ankle' in unified_lm and 'heel' in unified_lm and 'toe' in unified_lm:
                a = tuple(map(int, unified_lm['ankle']))
                h = tuple(map(int, unified_lm['heel']))
                t = tuple(map(int, unified_lm['toe']))
                
                pts = np.array([a, h, t], np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
                cv2.fillPoly(frame, [pts], (0, 100, 100))

            # 2. Red Dots
            for k, v in clean_lm.items():
                draw_it = False
                if 'nose' in k or 'eye' in k: draw_it = True
                elif current_side in k: draw_it = True
                
                if draw_it:
                    try:
                        pt = tuple(map(int, v))
                        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
                    except: pass
            
            # 3. Main Skeleton
            main_skel = [('shoulder', 'elbow'), ('elbow', 'wrist'), 
                         ('shoulder', 'hip'), ('hip', 'knee'), ('knee', 'ankle')]
            for k1, k2 in main_skel:
                if k1 in unified_lm and k2 in unified_lm:
                    p1 = tuple(map(int, unified_lm[k1]))
                    p2 = tuple(map(int, unified_lm[k2]))
                    cv2.line(frame, p1, p2, (0, 255, 255), 4, cv2.LINE_AA)

            # 4. Arcs
            if 'knee' in angles:
                 p1, p2, p3 = tuple(map(int, unified_lm['hip'])), tuple(map(int, unified_lm['knee'])), tuple(map(int, unified_lm['ankle']))
                 core.draw_angle_arc(frame, p1, p2, p3, angles['knee'], (0, 255, 0))

        if out: out.write(frame)
        cv2.imshow('NintAi Quad (MP)', frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()
    
    if not frames_data: sys.exit()
    
    df = pd.DataFrame([f['angles'] for f in frames_data])
    df['frame_idx'] = [f['frame_idx'] for f in frames_data]
    df = df[df['knee'] > 0]
    if df.empty: sys.exit()
    
    # --- Report Prep ---
    idx_bdc = df['knee'].idxmax()
    vals_bdc = df.loc[idx_bdc]
    idx_tdc = df['knee'].idxmin()
    vals_tdc = df.loc[idx_tdc]
    
    facing_right = True
    if frames_data[0]['side'] == 'left': facing_right = False
    
    best_x = -1e9 if facing_right else 1e9
    idx_front = 0
    
    target_k = 'toe'
    
    for i, f in enumerate(frames_data):
        if target_k in f['landmarks']:
            x = f['landmarks'][target_k][0]
            if facing_right:
                if x > best_x: 
                    best_x = x
                    idx_front = i
            else:
                if x < best_x:
                    best_x = x
                    idx_front = i
    vals_front = df.iloc[idx_front]

    stats = {
        'knee_ext_max': df['knee'].max(),
        'knee_flex_min': df['knee'].min(),
        'hip_closed_min': df['hip'].min(),
        'back_avg': df['back'].mean(),
        'back_range': df['back'].max() - df['back'].min(),
        'arm_avg': df['arm_torso'].mean(),
        'neck_avg': df['neck'].mean(),
        'wrist_tilt_avg': df['wrist_tilt'].mean(),
        'foot_angle_avg': df.get('foot_angle', pd.Series([0])).mean()
    }
    
    print("Generating Quad-View Snapshots (MP Tasks)...")
    
    def create_snapshot(idx, filename, title, overlay_metrics):
        c = cv2.VideoCapture(args.input)
        c.set(cv2.CAP_PROP_POS_FRAMES, frames_data[idx]['frame_idx']-1)
        _, img = c.read()
        c.release()
        if img is None: return None
        
        lm = frames_data[idx]['landmarks']
        clean_lm = frames_data[idx]['clean_lm']

        # Shoe
        if 'ankle' in lm and 'heel' in lm and 'toe' in lm:
             a = tuple(map(int, lm['ankle']))
             h = tuple(map(int, lm['heel']))
             t = tuple(map(int, lm['toe']))
             pts = np.array([a, h, t], np.int32)
             cv2.polylines(img, [pts], True, (0,255,255), 2)
             cv2.fillPoly(img, [pts], (0,100,100))

        # Skeleton
        skel = [('shoulder', 'elbow'), ('elbow', 'wrist'), 
                 ('shoulder', 'hip'), ('hip', 'knee'), ('knee', 'ankle')]
        for k1, k2 in skel:
             if k1 in lm and k2 in lm:
                 p1, p2 = tuple(map(int, lm[k1])), tuple(map(int, lm[k2]))
                 cv2.line(img, p1, p2, (0,255,255), 4, cv2.LINE_AA)
        
        # Dots
        for k, v in clean_lm.items():
            if frames_data[idx]['side'] in k or 'nose' in k:
                try: cv2.circle(img, tuple(map(int, v)), 6, (0,0,255), -1)
                except: pass

        cv2.putText(img, title, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        y = 100
        for k, v in overlay_metrics.items():
            label = f"{k}: {v:.1f}"
            cv2.putText(img, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            y += 40
            
        path = os.path.join(report_dir, filename)
        cv2.imwrite(path, img)
        return path

    snap_tdc = create_snapshot(df.index.get_loc(idx_tdc), "quad_tdc.jpg", "Top (TDC)", 
                               {'Knee Flex': vals_tdc['knee'], 'Hip Closed': vals_tdc['hip']})
    snap_bdc = create_snapshot(df.index.get_loc(idx_bdc), "quad_bdc.jpg", "Bottom (BDC)", 
                               {'Knee Ext': vals_bdc['knee'], 'Hip Open': vals_bdc['hip']})
    snap_front = create_snapshot(idx_front, "quad_front.jpg", "Front (Power)", 
                                 {'Knee': vals_front['knee'], 'Foot Ang': vals_front.get('foot_angle',0)})
    snap_over = create_snapshot(df.index.get_loc(idx_bdc), "quad_overall.jpg", "Overall Position", 
                                {'Back': stats['back_avg'], 'Neck': stats['neck_avg'], 'Foot (Avg)': stats['foot_angle_avg']})
    
    clinical_data = {'stats': stats}
    ai_text = ai_report.generate_ai_analysis(stats, [], api_key=args.api_key)
    
    print("Generating Clean Report (MP Tasks)...")
    report.generate_quad_report(snap_tdc, snap_bdc, snap_front, snap_over, clinical_data, args.output_excel.replace(".xlsx", ".pdf"), ai_text)
    print("Done")

if __name__ == "__main__":
    main()
