import cv2
from ultralytics import YOLO
import time

# Load the YOLOv8 model for object detection
model = YOLO('best.pt', task='detect')

# Export the model
model.export(format='openvino')  # creates 'best_openvino_model/'

# Load the exported OpenVINO model
ov_model = YOLO('best_openvino_model/', task='detect')

# Open the webcam
cap = cv2.VideoCapture(1)  # Modify the URL format

# Check if the webcam is opened successfully
if not cap.isOpened():
    raise RuntimeError("Failed to open webcam")

try:
    # Loop through the frames
    while True:
        # Read a frame
        success, frame = cap.read()

        if success:
            start_time = time.perf_counter()

            # Run YOLOv8 inference on the frame
            results = ov_model.predict(frame, conf=0.6)

            end_time = time.perf_counter()
            total_time = end_time - start_time
            fps = 1 / total_time

            # Visualise the results on the frame
            annotated_frame = results[0].plot()

            # Assuming class_names is a list of class names
            class_names = ['green', 'nongreen']  # Modify this according to your actual class names

            # Count the number of "green" and "non-green" tomatoes
            green_tomato_count = sum(1 for result in results for box in result.boxes if
                                     class_names[int(box.cls.item())] == 'green')
            non_green_tomato_count = sum(1 for result in results for box in result.boxes if
                                         class_names[int(box.cls.item())] == 'nongreen')

            # Calculate the total count
            total_tomato_count = green_tomato_count + non_green_tomato_count

            # Display the counts
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Green: {green_tomato_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Non-Green: {non_green_tomato_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Total: {total_tomato_count}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed or the window is closed
            if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty("YOLOv8 Inference",
                                                                           cv2.WND_PROP_VISIBLE) < 1:
                break
        else:
            # Break the loop if there are no more frames
            break
finally:
    # Release the video capture object and close the display windows using context managers
    cap.release()
    cv2.destroyAllWindows()
