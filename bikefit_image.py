import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True)

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First Point
    b = np.array(b)  # Mid Point
    c = np.array(c)  # End Point

    # Calculate angle
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to get feedback based on knee angle and elbow angle, and give saddle & elbow height recommendation
def get_feedback_and_adjustment(knee_angle, elbow_angle):
    feedback = ""
    saddle_adjustment = 0
    elbow_adjustment = 0
    adjustment_direction_saddle = ""
    adjustment_direction_elbow = ""

    if knee_angle < 25:
        feedback += "Knee angle is too small. Consider raising the saddle.\n"
        saddle_adjustment = (30 - knee_angle) * 0.2  # Estimated adjustment per degree
        adjustment_direction_saddle = "higher"
    elif knee_angle > 40:
        feedback += "Knee angle is too large. Consider lowering the saddle.\n"
        saddle_adjustment = (knee_angle - 35) * 0.2  # Estimated adjustment per degree
        adjustment_direction_saddle = "lower"
    else:
        feedback += "Knee angle is within an optimal range.\n"
        adjustment_direction_saddle = "no adjustment needed"

    if elbow_angle < 90:
        feedback += "Elbow angle is too acute. Consider raising the handlebars.\n"
        elbow_adjustment = (90 - elbow_angle) * 0.1  # Estimated adjustment per degree
        adjustment_direction_elbow = "higher"
    elif elbow_angle > 120:
        feedback += "Elbow angle is too open. Consider lowering the handlebars.\n"
        elbow_adjustment = (elbow_angle - 120) * 0.1  # Estimated adjustment per degree
        adjustment_direction_elbow = "lower"
    else:
        feedback += "Elbow angle is within an optimal range.\n"
        adjustment_direction_elbow = "no adjustment needed"

    return feedback, saddle_adjustment, adjustment_direction_saddle, elbow_adjustment, adjustment_direction_elbow

# Load an image of a cyclist
image_path = '/Users/pruthviomkargeedh/Desktop/BIKEFIT/test image/3.webp'  # Replace with the path to your image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Could not load image at {image_path}. Please check the file path.")
    exit()

# Convert the image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image to find pose landmarks
results = pose.process(image_rgb)

# Extract landmarks and calculate angles if pose detected
if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark

    # Get important joints
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image.shape[1], 
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image.shape[0]]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image.shape[1], 
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image.shape[0]]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image.shape[1], 
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image.shape[0]]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * image.shape[1], 
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image.shape[0]]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image.shape[1], 
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image.shape[0]]
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image.shape[1], 
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image.shape[0]]

    # Calculate angles
    knee_angle = calculate_angle(hip, knee, ankle)
    hip_angle = calculate_angle(shoulder, hip, knee)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    # Get feedback and adjustment recommendations
    feedback, saddle_adjustment, adjustment_direction_saddle, elbow_adjustment, adjustment_direction_elbow = get_feedback_and_adjustment(knee_angle, elbow_angle)

    # Draw the landmarks and angles on the image
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Annotate the image with angles and feedback
    cv2.putText(image, f'Knee Angle: {int(knee_angle)} degrees', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(image, f'Hip Angle: {int(hip_angle)} degrees', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(image, f'Elbow Angle: {int(elbow_angle)} degrees', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the feedback at the bottom
    y_position = 120
    for line in feedback.split('\n'):
        cv2.putText(image, line, (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_position += 30

    # Show saddle and elbow adjustments
    cv2.putText(image, f"Saddle Adjustment: {abs(saddle_adjustment):.2f} cm {adjustment_direction_saddle}", (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    y_position += 40
    cv2.putText(image, f"Elbow Adjustment: {abs(elbow_adjustment):.2f} cm {adjustment_direction_elbow}", (10, y_position),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the image
    cv2.imshow('Bike Fitting Assistant - Image Analysis', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No pose detected in the image. Please try a different image.")


