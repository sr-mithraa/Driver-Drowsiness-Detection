# Instead of live feed, working on video clips detection
# DISTRACTION

import cv2
import math
from ultralytics import YOLO

model = YOLO(
    r"/Users/mithra_sr/PycharmProjects/DrowsinessDetection/v8m_drowsy_detect_model.pt")

# Open webcam or video
video_path = 0  # Use 0 for live webcam, or provide a video path
cap = cv2.VideoCapture(video_path)


# Camera Placement (Option          qs: "front", "top-right", "top-left", "dashboard")
CAMERA_POSITION = "top-right"


# Thresholds
FPS = 12
CLOSED_EYE_THRESHOLD = FPS  # 1 sec
YAWN_THRESHOLD = FPS * 1.5  # 1.5 sec
MOUTH_OPEN_RATIO = 0.8  # Threshold for "wide open" mouth
TILT_THRESHOLD = 40  # Degrees threshold for detecting tilt
MOVE_THRESHOLD = 30  # Pixels threshold for detecting face movement


# Counters
consecutive_closed_eye_count = 0
consecutive_yawn_count = 0


# Initial face position (for center calibration)
center_ref_x, center_ref_y = None, None


def calculate_angle(pt1, pt2):
    """Calculate the angle between two points."""
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    return math.degrees(math.atan2(dy, dx))


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO detection
    # Detect Face, Eyes, Mouth
    results = model.predict(frame, classes=[0, 1, 2, 3, 4, 5], conf=0.5)

    eye_closed_detected = False
    mouth_open_detected = False
    left_eye, right_eye, face_center = None, None, None
    head_tilt = "Straight"
    face_position = "Centered"

    # Process results
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_id in [2, 4]:  # Eye closed
                eye_closed_detected = True
            if class_id in [1, 2]:  # Left Eye Open or Closed
                left_eye = ((x1 + x2) // 2, (y1 + y2) // 2)
            if class_id in [3, 4]:  # Right Eye Open or Closed
                right_eye = ((x1 + x2) // 2, (y1 + y2) // 2)
            if class_id == 5:  # Mouth detected
                mouth_width = x2 - x1
                mouth_height = y2 - y1
                if (mouth_height / mouth_width) > MOUTH_OPEN_RATIO:
                    mouth_open_detected = True
            if class_id == 0:  # Face detected
                face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if center_ref_x is None or center_ref_y is None:
                    center_ref_x, center_ref_y = face_center

    # Detect Head Tilt
    if left_eye and right_eye:
        angle = calculate_angle(left_eye, right_eye)
        if angle > TILT_THRESHOLD:
            head_tilt = "Head Tilted Right"
        elif angle < -TILT_THRESHOLD:
            head_tilt = "Head Tilted Left"

    # Detect Face Position relative to the reference center
    if face_center:
        delta_x = face_center[0] - center_ref_x
        delta_y = face_center[1] - center_ref_y
        if delta_x > MOVE_THRESHOLD:
            face_position = "Looking Right"
        elif delta_x < -MOVE_THRESHOLD:
            face_position = "Looking Left"
        elif delta_y > MOVE_THRESHOLD:
            face_position = "Looking Down"
        elif delta_y < -MOVE_THRESHOLD:
            face_position = "Looking Up"

    # Update counters
    consecutive_closed_eye_count = consecutive_closed_eye_count + \
        1 if eye_closed_detected else 0
    consecutive_yawn_count = consecutive_yawn_count + 1 if mouth_open_detected else 0

    # Annotate frame
    annotated_frame = results[0].plot()
    annotated_frame = cv2.resize(annotated_frame, (1200, 640))

    # Display info
    cv2.putText(annotated_frame, f"Face Position: {face_position}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
    cv2.putText(annotated_frame, f"Closed Eye Frames: {consecutive_closed_eye_count}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(annotated_frame, f"Yawn Frames: {consecutive_yawn_count}", (50, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Trigger Alerts
    if consecutive_closed_eye_count >= CLOSED_EYE_THRESHOLD:
        cv2.putText(annotated_frame, "ALERT! Drowsiness Detected", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    if consecutive_yawn_count >= YAWN_THRESHOLD:
        cv2.putText(annotated_frame, "WARNING! Yawning Detected", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
    if face_position in ["Looking Left", "Looking Right"]:
        cv2.putText(annotated_frame, "WARNING! Driver Looking Sideways", (50, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    if face_position in ["Looking Up", "Looking Down"]:
        cv2.putText(annotated_frame, "WARNING! Face Position Abnormal", (50, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 165, 0), 3)

    # Display frame
    cv2.imshow("YOLO Driver Monitoring", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()
