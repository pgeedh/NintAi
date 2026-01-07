from ultralytics import YOLO
import cv2
import numpy as np

class PoseDetector:
    def __init__(self, model_path='yolo11n-pose.pt'):
        """
        Initializes the YOLO11 Pose model.
        """
        self.model = YOLO(model_path)

    def predict(self, image):
        """
        Runs YOLO11 pose estimation on the frame.
        """
        results = self.model(image, verbose=False)
        return results[0] if results else None

    def get_landmarks_dict(self, results, image_shape):
        """
        Extracts ALL COCO keypoints (0-16) and maps them to named keys (left_*, right_*).
        """
        h, w = image_shape[:2]
        lm_dict = {}
        
        if results.keypoints is None or len(results.keypoints) == 0:
            return lm_dict

        # Take first person
        kpts = results.keypoints.data[0].cpu().numpy()
        
        # COCO Mapping
        # 0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear
        # 5: L-Shoulder, 6: R-Shoulder, 7: L-Elbow, 8: R-Elbow
        # 9: L-Wrist, 10: R-Wrist, 11: L-Hip, 12: R-Hip
        # 13: L-Knee, 14: R-Knee, 15: L-Ankle, 16: R-Ankle
        
        mapping = {
            0: 'nose',
            1: 'left_eye', 2: 'right_eye',
            3: 'left_ear', 4: 'right_ear',
            5: 'left_shoulder', 6: 'right_shoulder',
            7: 'left_elbow', 8: 'right_elbow',
            9: 'left_wrist', 10: 'right_wrist',
            11: 'left_hip', 12: 'right_hip',
            13: 'left_knee', 14: 'right_knee',
            15: 'left_ankle', 16: 'right_ankle'
        }
        
        for idx, name in mapping.items():
            if idx < len(kpts):
                x, y, conf = kpts[idx]
                if conf > 0.3: # Threshold
                    lm_dict[name] = [x, y]
        
        # Backward compatibility for 'simple' names (default to Left for now, but analysis should pick side)
        # Actually, let's NOT default to left here to avoid confusion. 
        # The analyzer should map 'knee' -> 'left_knee' or 'right_knee' based on detection.
        # But to keep existing code (core.py) running without crash, we might map detected side?
        # Let's map strict Left for legacy parts IF they exist.
        for simple in ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle']:
            left_key = f"left_{simple}"
            if left_key in lm_dict:
                lm_dict[simple] = lm_dict[left_key]

        return lm_dict
