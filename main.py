import cv2
import numpy as np
import csv
import os
from inference_sdk import InferenceHTTPClient
import easyocr
from ultralytics import YOLO

# Initialize the InferenceHTTPClient for license plate detection
PLATE_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="jJmY7yIyUC38f3rUzSSk"
)

# Initialize the InferenceHTTPClient for car detection
CAR_CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="jJmY7yIyUC38f3rUzSSk"
)

# Load the pre-trained license plate detector model
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'

# Initialize YOLO model for license plate text recognition
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Open the video file
video_path = "WhatsApp Video 2024-10-23 at 03.29.25_2ab6ad1e.mp4"
cap = cv2.VideoCapture(video_path)

# Prepare CSV file
csv_file_path = "C:/Users/sansa/Downloads/inference_results.csv"
csv_columns = ['frame_number', 'x_car', 'y_car', 'car_confidence', 
               'x_numplate', 'y_numplate', 'width_numplate', 'height_numplate', 
               'class', 'numplate_confidence', 'color', 'num_plate']
csv_data = []  # List to accumulate inference data

# Directory for saving cropped license plate images
output_image_dir = "C:/Users/sansa/Downloads/cropped_numplates/"
os.makedirs(output_image_dir, exist_ok=True)

# Output video settings
output_video_path = "C:/Users/sansa/Downloads/output_with_inference.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_counter = 0  # Track frame number
skip_frames = 11    # Number of frames to skip between inferences

# Function to get the color name based on average BGR values
def get_color_name(bgr):
    blue, green, red = bgr
    if red > 200 and green > 200 and blue > 200:
        return "White"
    elif red < 100 and green < 100 and blue < 100:
        return "Black"
    elif 100 < red < 150 and 100 < green < 150 and 100 < blue < 150:
        return "Gray"
    elif red > 150 and green < 100 and blue < 100:
        return "Red"
    elif green > 150 and red < 100 and blue < 100:
        return "Green"
    elif blue > 150 and red < 100 and green < 100:
        return "Blue"
    elif red > 200 and green > 200 and blue < 100:
        return "Yellow"
    else:
        return "Unknown"

# Function to calculate the average color within the bounding box
def get_average_color(frame, box):
    x, y, w, h = box
    crop = frame[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
    avg_color = cv2.mean(crop)[:3]
    return avg_color

# Function to read the license plate text using EasyOCR
def read_license_plate(cropped_image):
    """Reads the text from the cropped license plate image."""
    detections = reader.readtext(cropped_image)
    
    if len(detections) == 0:
        return "No text detected"
    
    # Combine the detected texts
    plate_text = " ".join([result[1] for result in detections])
    return plate_text

# Process frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames
    if frame_counter % (skip_frames + 1) != 0:
        frame_counter += 1
        continue

    # Save the current frame as an image for inference
    frame_path = "C:/Users/sansa/Downloads/frame.jpg"
    cv2.imwrite(frame_path, frame)

    # Perform car detection on the frame
    car_result = CAR_CLIENT.infer(frame_path, model_id="vehiclecount/4")

    # Process car detection results
    if 'predictions' in car_result:
        for prediction in car_result['predictions']:
            x_car, y_car, w_car, h_car = prediction['x'], prediction['y'], prediction['height'], prediction['width']
            car_confidence = prediction['confidence']
            avg_color_bgr = get_average_color(frame, (x_car, y_car, w_car, h_car))
            color_name = get_color_name(avg_color_bgr)

            # Perform license plate detection on the same frame
            plate_result = PLATE_CLIENT.infer(frame_path, model_id="vehicle-registration-plates-trudk/2")
            
            # Process license plate detection results
            if 'predictions' in plate_result:
                for plate_prediction in plate_result['predictions']:
                    x_numplate = plate_prediction['x']
                    y_numplate = plate_prediction['y']
                    width_numplate = plate_prediction['width']
                    height_numplate = plate_prediction['height']
                    numplate_confidence = plate_prediction['confidence']
                    plate_class = plate_prediction['class']

                    # Crop the license plate region for OCR
                    plate_roi = frame[int(y_numplate - height_numplate / 2):int(y_numplate + height_numplate / 2), 
                                      int(x_numplate - width_numplate / 2):int(x_numplate + width_numplate / 2)]

                    # Save the cropped license plate image for reference
                    cropped_image_path = os.path.join(output_image_dir, f"frame_{frame_counter}_plate_cropped.jpg")
                    cv2.imwrite(cropped_image_path, plate_roi)

                    # Read the license plate number using EasyOCR
                    num_plate = read_license_plate(plate_roi)

                    # Print predictions to terminal
                    print(f"Frame: {frame_counter}, Car Confidence: {car_confidence:.2f}, License Plate: {num_plate}")

                    # Store all relevant data for the current instance
                    csv_data.append({
                        'frame_number': frame_counter,
                        'x_car': x_car,
                        'y_car': y_car,
                        'car_confidence': car_confidence,
                        'x_numplate': x_numplate,
                        'y_numplate': y_numplate,
                        'width_numplate': width_numplate,
                        'height_numplate': height_numplate,
                        'class': plate_class,
                        'numplate_confidence': numplate_confidence,
                        'color': color_name,
                        'num_plate': num_plate
                    })

                    # Draw the car bounding box and color label on the frame
                    top_left_car = (int(x_car - w_car / 2), int(y_car - h_car / 2))
                    bottom_right_car = (int(x_car + w_car / 2), int(y_car + h_car / 2))
                    cv2.rectangle(frame, top_left_car, bottom_right_car, (0, 255, 0), 2)  # Green box for car
                    cv2.putText(frame, f"{color_name} {car_confidence:.2f}", 
                                (top_left_car[0], top_left_car[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Draw the license plate bounding box and text on the frame
                    top_left_plate = (int(x_numplate - width_numplate / 2), int(y_numplate - height_numplate / 2))
                    bottom_right_plate = (int(x_numplate + width_numplate / 2), int(y_numplate + height_numplate / 2))
                    cv2.rectangle(frame, top_left_plate, bottom_right_plate, (255, 0, 0), 2)  # Blue box for license plate
                    cv2.putText(frame, f"Plate: {num_plate}", 
                                (top_left_plate[0], top_left_plate[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Write the processed frame to the output video
    out_video.write(frame)


    # Display the processed frame in a window
    cv2.imshow('Processed Video', frame)

    frame_counter += 1

    # You can add this if you want to break processing early by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video resources
cap.release()
out_video.release()

# Write the CSV data to the output file
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(csv_data)

print(f"Results saved to {csv_file_path}")