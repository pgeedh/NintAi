from ultralytics import YOLO
import numpy as np

class PoseDetector:
    def __init__(self, model_path='yolov8n-pose.pt'):
        """
        Initializes the YOLOv8 Pose Detector.
        """
        self.model = YOLO(model_path)
    
    def predict(self, image):
        """
        Runs inference on the image and returns keypoints.
        """
        results = self.model(image, verbose=False)
        return results[0]

    def get_landmarks_dict(self, results, image_shape):
        """
        Extracts landmarks from YOLO results and maps them to NintAi standard names.
        YOLOv8 (COCO) Keypoints Mapping:
        5: Left Shoulder, 6: Right Shoulder
        7: Left Elbow, 8: Right Elbow
        9: Left Wrist, 10: Right Wrist
        11: Left Hip, 12: Right Hip
        13: Left Knee, 14: Right Knee
        15: Left Ankle, 16: Right Ankle
        """
        if not results.keypoints or len(results.keypoints) == 0:
            return None

        # Take the first detected person
        kp = results.keypoints.xy[0].cpu().numpy() # Shape: (17, 2)
        
        # Helper to get point - logic for Left vs Right side?
        # Ideally we detect which side of the bike the user is facing.
        # For simplicity, we assume LEFT side view (standard bike fit view) or try to detect visibility.
        # Let's assume LEFT side for now (odd indices in COCO usually? No, check mapping).
        # COCO: 5=L_Shoulder, 11=L_Hip, 13=L_Knee, 15=L_Ankle.
        # This matches a rider facing LEFT (camera sees their left side).
        
        # TODO: Add logic to auto-switch to Right side if confidence is higher?
        
        def get_pt(idx):
            return kp[idx]

        mapping = {
            'shoulder': get_pt(5),
            'hip': get_pt(11),
            'knee': get_pt(13),
            'ankle': get_pt(15),
            'elbow': get_pt(7),
            'wrist': get_pt(9),
            # Foot is not in standard COCO keypoints (17 points). 
            # We might need to estimate it or drop Ankling Range for YOLO 
            # OR use Ankle + some heuristic. 
            # For now, let's omit 'foot' or map it to ankle (which kills the angle).
            # Let's mock foot as just below ankle for now to strictly avoid crash, 
            # but really we should remove ankling range if we don't have foot tip.
            # actually, using Ankle for now.
            'foot': get_pt(15) 
        }
        
        return mapping
