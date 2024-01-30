# Libraries Imported
import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO
from sort.sort import *

# Initialize video capture with specified settings
cap = cv2.VideoCapture("input/videos/testVideo01.mp4")
cap.set(3, 1280)
cap.set(4, 720)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Load YOLO model for object detection
model = YOLO("yolov8n.pt")

# Define object classes recognized by YOLO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Load region mask for processing
mask = cv2.imread("input/images/testVideo01Mask.png")

# Initialize the SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define the region for counting objects
limits = [390, 600, 750, 600]

# Counting variable
totalCount = []

# Video writer for output
video_writer = cv2.VideoWriter("output/object_counting_output.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

# Main processing loop
while True:
    # Read a frame from the video
    success, img = cap.read()
    
    # Apply region mask to the frame
    imgRegion = cv2.bitwise_and(img, mask)
    
    # Detect objects in the region using YOLO
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    # Process YOLO results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Extract confidence and class information
            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Check if the detected object is a car, motorbike, bus, or truck with confidence > 0.4
            if currentClass in ['car', 'motorbike', 'bus', 'truck'] and conf > 0.4:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update tracker with object detections
    resultsTracker = tracker.update(detections)

    # Draw line/region on the frame
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 0, 255), thickness=5)

    # Process tracked results and update count
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1

        # Draw bounding box and centroid
        cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=2, colorR=(255, 0, 0))
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Check if the object crosses the counting region
        if limits[0] < cx < limits[2] and limits[1] - 10 < cy < limits[1] + 10:
            if id not in totalCount:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 255, 0), thickness=5)

    # Display count on the frame
    cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50, 50))

    # Show processed frame and write to output video
    cv2.imshow("image", img)
    video_writer.write(img)

    # Quit when 'q' key is pressed
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()