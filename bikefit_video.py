import cv2
import numpy as np
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False)

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

# Function to get feedback based on knee and elbow angles and give saddle & elbow height recommendation
def get_feedback_and_adjustment(knee_angle_avg, elbow_angle_avg):
    feedback = ""
    saddle_adjustment = 0
    elbow_adjustment = 0
    saddle_direction = ""
    elbow_direction = ""
    
    # Adjust Saddle based on Knee Angle
    if knee_angle_avg < 25:
        feedback += "Knee angle is too small. Consider raising the saddle. "
        saddle_adjustment = (30 - knee_angle_avg) * 0.2  # Estimated adjustment per degree
        saddle_direction = "higher"
    elif knee_angle_avg > 40:
        feedback += "Knee angle is too large. Consider lowering the saddle. "
        saddle_adjustment = (knee_angle_avg - 35) * 0.2  # Estimated adjustment per degree
        saddle_direction = "lower"
    else:
        feedback += "Knee angle is within an optimal range. "

    # Adjust Elbow Height based on Elbow Angle
    if elbow_angle_avg < 80:
        feedback += "Elbow angle is too small. Consider raising the arm pads. "
        elbow_adjustment = (85 - elbow_angle_avg) * 0.1  # Estimated adjustment per degree
        elbow_direction = "higher"
    elif elbow_angle_avg > 120:
        feedback += "Elbow angle is too large. Consider lowering the arm pads. "
        elbow_adjustment = (elbow_angle_avg - 115) * 0.1  # Estimated adjustment per degree
        elbow_direction = "lower"
    else:
        feedback += "Elbow angle is within an optimal range. "

    return feedback, saddle_adjustment, saddle_direction, elbow_adjustment, elbow_direction

# Load video
cap = cv2.VideoCapture('/Users/pruthviomkargeedh/Desktop/bikefit/Test Video/testvideo2.mp4')

# Data collection lists
time_steps = []
knee_angles = []
hip_angles = []
elbow_angles = []
ankling_ranges = []

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Convert the image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to find pose landmarks
    results = pose.process(image_rgb)

    # Extract landmarks and calculate angles
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get important joints
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1], 
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame.shape[1], 
               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame.shape[0]]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * frame.shape[1], 
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * frame.shape[0]]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * frame.shape[1], 
                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame.shape[0]]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame.shape[1], 
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame.shape[0]]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame.shape[1], 
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame.shape[0]]
        foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * frame.shape[0]]

        # Calculate angles
        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        ankling_range = calculate_angle(knee, ankle, foot)

        # Draw the landmarks and angles on the image
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Annotate the frame with angles
        cv2.putText(frame, f'Knee: {int(knee_angle)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f'Hip: {int(hip_angle)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f'Elbow: {int(elbow_angle)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f'Ankle: {int(ankling_range)}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Append the angles and timestep
        time_steps.append(frame_count)
        knee_angles.append(knee_angle)
        hip_angles.append(hip_angle)
        elbow_angles.append(elbow_angle)
        ankling_ranges.append(ankling_range)

    # Display the annotated frame
    cv2.imshow('Bike Fitting Assistant', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()



# Calculate average knee and elbow angles and recommend adjustments
average_knee_angle = np.mean(knee_angles)
average_elbow_angle = np.mean(elbow_angles)
feedback, saddle_adjustment, saddle_direction, elbow_adjustment, elbow_direction = get_feedback_and_adjustment(average_knee_angle, average_elbow_angle)

# Convert data to a dataframe and save to Excel
angles_df = pd.DataFrame({
    'Time Step': time_steps,
    'Knee Angle': knee_angles,
    'Hip Angle': hip_angles,
    'Elbow Angle': elbow_angles,
    'Ankling Range': ankling_ranges
})
angles_df.to_excel('/Users/pruthviomkargeedh/Desktop/bikefit/testdata/angles_data.xlsx', index=False)

# Display the final saddle and elbow adjustment feedback
print(f"Average Knee Angle: {average_knee_angle:.2f} degrees")
print(f"Average Elbow Angle: {average_elbow_angle:.2f} degrees")
print(f"Saddle Adjustment Recommendation: {abs(saddle_adjustment):.2f} cm {saddle_direction}")
print(f"Elbow Adjustment Recommendation: {abs(elbow_adjustment):.2f} cm {elbow_direction}")

# Display the final recommendation at the end of the video
end_frame = np.zeros((500, 800, 3), dtype=np.uint8)
cv2.putText(end_frame, "Adjustment Recommendations:", (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
cv2.putText(end_frame, f"Saddle Height: {abs(saddle_adjustment):.2f} cm {saddle_direction}", (20, 250),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(end_frame, f"Elbow Height: {abs(elbow_adjustment):.2f} cm {elbow_direction}", (20, 350),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imshow('Bike Fitting Assistant - Final Recommendation', end_frame)
cv2.waitKey(0)







