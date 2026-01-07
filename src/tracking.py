from ultralytics import YOLO
import cv2

class PoseDetector:
    def __init__(self, model_path='yolo11n-pose.pt'):
        """
        Initializes the YOLO11 Pose model.
        Args:
            model_path (str): Path to the YOLO11 pose model. 
                              Defaults to 'yolo11n-pose.pt' (Nano) for speed.
                              Requires 'pip install ultralytics'
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
        Extracts keypoints and normalizes them into a dictionary format compatible with NintAi's core.
        YOLOv8/11 COCO Keypoints:
        0: Nose, 1: Eye, ..., 5: Shoulder, 7: Elbow, 9: Wrist, 11: Hip, 13: Knee, 15: Ankle
        (Left side odd, Right side even... logic handles mainly left side fit for now)
        """
        h, w = image_shape[:2]
        lm_dict = {}
        
        if results.keypoints is None or len(results.keypoints) == 0:
            return lm_dict

        # Assuming single person - take the FIRST detection
        # data format: [x, y, conf]
        kpts = results.keypoints.data[0].cpu().numpy()
        
        # Mapping COCO keypoint indices to NintAi names
        # Assuming cyclist is facing left (showing left side) -> using odd indices
        # If confidence is low, could check right side? For now strict left logic.
        
        mapping = {
            5: 'shoulder',  # Left Shoulder
            7: 'elbow',     # Left Elbow
            9: 'wrist',     # Left Wrist
            11: 'hip',      # Left Hip
            13: 'knee',     # Left Knee
            15: 'ankle'     # Left Ankle
        }
        
        # Heuristic for 'foot':
        # YOLO doesn't have a 'toe' or 'metatarsal' point. 
        # We can approximate 'foot' by extending line from knee->ankle? 
        # Or just use Ankle for now.
        # NintAi core uses 'foot' for ankling. 
        # Let's map 'foot' to 'ankle' (index 15) for safety so it doesn't crash, 
        # but ankling logic will be 0.
        
        for idx, name in mapping.items():
            if idx < len(kpts):
                x, y, conf = kpts[idx]
                if conf > 0.3: # Threshold
                    lm_dict[name] = [x, y]
        
        # Add 'foot' as ankle copy if ankle exists
        if 'ankle' in lm_dict:
            lm_dict['foot'] = lm_dict['ankle']

        return lm_dict
