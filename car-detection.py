import cv2
import numpy as np
import dlib
from collections import OrderedDict
from pykalman import KalmanFilter
import math
import csv
import os
# Load YOLOv4 model


def load_yolo_model(config_file, weights_file):
    net = cv2.dnn.readNet(weights_file, config_file)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1]
                     for i in net.getUnconnectedOutLayers().flatten()]
    return net, output_layers

# Process the frame and detect cars


def detect_cars(net, output_layers, frame, conf_threshold=0.5, nms_threshold=0.4):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame, 1/255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold and class_id == 2:  # class_id 2 corresponds to "car" in the COCO dataset
                center_x, center_y, w, h = (
                    detection[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, conf_threshold, nms_threshold)
    return [(boxes[i], confidences[i], class_ids[i]) for i in indices]

# Draw detected cars on the frame


def calculate_px_to_meter_ratio(reference_object_real_size, reference_object_pixel_size):
    """
    Calculates the pixel to meter ratio.

    Parameters:
    - reference_object_real_size: The real size of the reference object (in meters).
    - reference_object_pixel_size: The size of the reference object in the image (in pixels).

    Returns:
    - The pixel to meter ratio.
    """
    return reference_object_real_size / reference_object_pixel_size


def draw_cars(frame, detections, car_trackers, car_speeds):
    for car_id, ((x, y, w, h), _, _) in enumerate(detections):
        if car_id not in car_trackers:
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(x, y, x + w, y + h)
            tracker.start_track(frame, rect)
            car_trackers[car_id] = tracker
        else:
            tracker = car_trackers[car_id]
            tracker.update(frame)
            pos = tracker.get_position()
            x, y, w, h = int(pos.left()), int(pos.top()), int(
                pos.width()), int(pos.height())

        speed = car_speeds.get(car_id, "N/A")
        if isinstance(speed, float):
            label = f"Car {car_id}: {speed:.2f} km/h"
        else:
            label = f"Car {car_id}: {speed}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Main function
def main(config_file, weights_file):
    net, output_layers = load_yolo_model(config_file, weights_file)

    video_dir = "/Users/dilarahalavurt/Downloads/archive (3)/video/"

    for filename in os.listdir(video_dir):
        # adjust this if your videos are in a different format
        if filename.endswith(".avi"):
            video_file = os.path.join(video_dir, filename)
            print(f"Processing {video_file}...")
            cap = cv2.VideoCapture(video_file)

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = 0
            car_trackers = OrderedDict()
            car_speeds = {}
            car_count = {}
            # A simple estimation, you can adjust this value based on real-world measurements

            while cap.isOpened():

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                detections = detect_cars(net, output_layers, frame)

                for car_id, ((x, y, w, h), _, _) in enumerate(detections):
                    if car_id not in car_count:
                        car_count[car_id] = True

                if frame_count % fps == 0:  # Update car speeds every second
                    updated_car_speeds = {}
                    for car_id, ((x, y, w, h), _, _) in enumerate(detections):
                        if car_id in car_trackers:
                            tracker = car_trackers[car_id]
                            pos = tracker.get_position()
                            x1, y1, w1, h1 = int(pos.left()), int(
                                pos.top()), int(pos.width()), int(pos.height())

                            x2, y2, w2, h2 = x, y, w, h
                            px_to_meter_ratio = calculate_px_to_meter_ratio(
                                1.85, math.sqrt((w*w)+(h*h)))
                            # Calculate distance in meters and time in seconds
                            dx = abs(x2 - x1) * px_to_meter_ratio
                            dy = abs(y2 - y1) * px_to_meter_ratio
                            # Euclidean distance
                            distance = math.sqrt(dx**2 + dy**2)
                            dt = 1

                            # Calculate speed in km/h
                            speed = (distance / dt) * 3.6
                            updated_car_speeds[car_id] = speed

                    car_speeds = updated_car_speeds
                draw_cars(frame, detections, car_trackers, car_speeds)

                cv2.imshow("Video", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            if car_speeds:
                avg_speed = sum(car_speeds.values()) / len(car_speeds)
                print(f"Average speed: {avg_speed:.2f} km/h")
            print(f"Total cars detected: {len(car_count)}")

            with open('results.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [filename, f"{avg_speed:.2f} km/h", len(car_count)])


if __name__ == "__main__":
    config_file = "/Users/dilarahalavurt/Desktop/ComputerVisionProject/yolov4.cfg"
    weights_file = "/Users/dilarahalavurt/Desktop/ComputerVisionProject/yolov4.weights"
    main(config_file, weights_file)
