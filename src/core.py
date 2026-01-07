import numpy as np

# --- Calibration Logic ---
class BikeCalibrator:
    def __init__(self):
        self.px_per_cm = None

    def calibrate(self, point_a, point_b, known_distance_cm):
        """
        Calibrates the scale based on two points and a real-world distance.
        """
        dist_px = np.linalg.norm(np.array(point_a) - np.array(point_b))
        if dist_px == 0:
            raise ValueError("Calibration points are identical.")
        self.px_per_cm = dist_px / known_distance_cm
        return self.px_per_cm

    def px_to_cm(self, pixels):
        if self.px_per_cm is None:
            return 0 # Or raise error
        return pixels / self.px_per_cm

# --- Geometry & Analysis ---

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points a, b, and c (where b is the vertex).
    Points should be (x, y) tuples or lists.
    Returns the angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def analyze_posture(landmarks_dict):
    """
    Calculates key angles for bike fitting from the landmarks dictionary.
    Returns a dictionary of angles.
    """
    angles = {}
    
    # Check if we have valid points (not [0,0] which YOLO might return for missing)
    def valid(pt):
        return not (pt[0] == 0 and pt[1] == 0)

    if valid(landmarks_dict['hip']) and valid(landmarks_dict['knee']) and valid(landmarks_dict['ankle']):
        angles['knee'] = calculate_angle(landmarks_dict['hip'], landmarks_dict['knee'], landmarks_dict['ankle'])
    else:
        angles['knee'] = 0

    if valid(landmarks_dict['shoulder']) and valid(landmarks_dict['hip']) and valid(landmarks_dict['knee']):
        angles['hip'] = calculate_angle(landmarks_dict['shoulder'], landmarks_dict['hip'], landmarks_dict['knee'])
    else:
        angles['hip'] = 0
    
    if valid(landmarks_dict['shoulder']) and valid(landmarks_dict['elbow']) and valid(landmarks_dict['wrist']):
        angles['elbow'] = calculate_angle(landmarks_dict['shoulder'], landmarks_dict['elbow'], landmarks_dict['wrist'])
    else:
        angles['elbow'] = 0

    # Ankling Range (Knee - Ankle - Foot)
    # Note: YOLO might not give foot, so check existence
    if 'foot' in landmarks_dict and valid(landmarks_dict['foot']) and valid(landmarks_dict['knee']) and valid(landmarks_dict['ankle']):
        angles['ankle'] = calculate_angle(landmarks_dict['knee'], landmarks_dict['ankle'], landmarks_dict['foot'])
    else:
        angles['ankle'] = 0

    return angles

def get_feedback(knee_angle, elbow_angle):
    """
    Generates feedback and adjustment recommendations based on knee and elbow angles.
    """
    feedback_lines = []
    saddle_adj = { 'value': 0, 'direction': 'ok' }
    elbow_adj = { 'value': 0, 'direction': 'ok' }

    # Knee Angle / Saddle Height Logic
    if knee_angle > 0: # Only analyze if valid
        if knee_angle < 25:
            feedback_lines.append("Knee angle too small (<25°). Saddle is likely too LOW.")
            saddle_adj['value'] = (30 - knee_angle) * 0.2
            saddle_adj['direction'] = "HIGHER"
        elif knee_angle > 40:
            feedback_lines.append("Knee angle too large (>40°). Saddle is likely too HIGH.")
            saddle_adj['value'] = (knee_angle - 35) * 0.2
            saddle_adj['direction'] = "LOWER"
        else:
            feedback_lines.append("Knee angle is optimal (25°-40°).")

    # Elbow Angle / Handlebar Height Logic
    if elbow_angle > 0:
        if elbow_angle < 90: 
            feedback_lines.append("Elbow angle too acute (<90°). Bars too LOW/CLOSE.")
            elbow_adj['value'] = (90 - elbow_angle) * 0.1
            elbow_adj['direction'] = "HIGHER/FURTHER"
        elif elbow_angle > 120: 
            feedback_lines.append("Elbow angle too open (>120°). Bars too HIGH/FAR.")
            elbow_adj['value'] = (elbow_angle - 120) * 0.1
            elbow_adj['direction'] = "LOWER/CLOSER"
        else:
            feedback_lines.append("Elbow angle is optimal (90°-120°).")

    return feedback_lines, saddle_adj, elbow_adj
