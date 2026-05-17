# YAWNING
import cv2
from ultralytics import YOLO


# Load the YOLO model
model = YOLO(
    r"/Users/mithra_sr/PycharmProjects/DrowsinessDetection/v8m_drowsy_detect_model.pt")


# Open webcam or video
video_path = 0  # Use 0 for live webcam or provide a video path
cap = cv2.VideoCapture(video_path)


# Frame rate (adjust based on camera)
FPS = 12
CLOSED_EYE_THRESHOLD = FPS  # 1 sec
YAWN_THRESHOLD = FPS * 1.5  # 1.5 sec
MOUTH_OPEN_RATIO = 0.8  # Threshold for "wide open" mouth


# Counters
consecutive_closed_eye_count = 0
consecutive_yawn_count = 0


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO detection
    # 2: Left Eye Closed, 4: Right Eye Closed, 5: Mouth
    results = model.predict(frame, classes=[2, 4, 5], conf=0.5)

    eye_closed_detected = False
    mouth_open_detected = False

    # Process results
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_id in [2, 4]:  # Eye closed
                eye_closed_detected = True

            if class_id == 5:  # Mouth detected
                mouth_width = x2 - x1
                mouth_height = y2 - y1

                # Check if mouth is open based on height-width ratio
                if (mouth_height / mouth_width) > MOUTH_OPEN_RATIO:
                    mouth_open_detected = True

    # Update counters
    consecutive_closed_eye_count = consecutive_closed_eye_count + \
        1 if eye_closed_detected else 0
    consecutive_yawn_count = consecutive_yawn_count + 1 if mouth_open_detected else 0

    # Annotate frame
    annotated_frame = results[0].plot()
    annotated_frame = cv2.resize(annotated_frame, (1200, 640))

    # Display info
    cv2.putText(annotated_frame, f"Closed Eye Frames: {consecutive_closed_eye_count}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(annotated_frame, f"Yawn Frames: {consecutive_yawn_count}", (50, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Trigger drowsiness alert
    if consecutive_closed_eye_count >= CLOSED_EYE_THRESHOLD:
        cv2.putText(annotated_frame, "ALERT! Drowsiness Detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Trigger yawning alert
    if consecutive_yawn_count >= YAWN_THRESHOLD:
        cv2.putText(annotated_frame, "WARNING! Yawning Detected", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    # Display frame
    cv2.imshow("YOLO Driver Monitoring", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()
