# DROWSY

from ultralytics import YOLO
import cv2
import os


# from model_test_modfy import results


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Load the YOLO model
model = YOLO(
    r"/Users/mithra_sr/PycharmProjects/DrowsinessDetection/v8m_drowsy_detect_model.pt")

# Open the webcam
video_path = 0  # Update with your video path
cap = cv2.VideoCapture(video_path)


# Frame rate of the webcam (Assuming 15 FPS, adjust based on your camera)
FPS = 12
# If eyes are closed for 1 second (15 frames), trigger alert
CLOSED_EYE_THRESHOLD = FPS


# Consecutive closed-eye frame counter
consecutive_closed_eye_count = 0


# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO prediction with confidence threshold
    results = model.predict(frame, classes=[2, 4], conf=0.5)

    eye_closed_detected = False  # Track if the eye is closed in the current frame

    # Process results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls.item())

            # Left eye closed (2) or Right eye closed (4)
            if class_id in [2, 4]:
                eye_closed_detected = True

    # If eyes are closed, increase the consecutive counter
    if eye_closed_detected:
        consecutive_closed_eye_count += 1
    else:
        consecutive_closed_eye_count = 0  # Reset counter if eyes are open

    # Annotate the frame
    annotated_frame = results[0].plot()
    annotated_frame = cv2.resize(annotated_frame, (1200, 640))

    # Display text on frame
    cv2.putText(annotated_frame, f"Closed Eye Frames: {consecutive_closed_eye_count}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Detect alert if eyes are closed for 1 second (15 frames)
    if consecutive_closed_eye_count >= CLOSED_EYE_THRESHOLD:
        cv2.putText(annotated_frame, "ALERT! Drowsiness Detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Display the annotated frame
    cv2.imshow("YOLO Eye Tracking", annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()
