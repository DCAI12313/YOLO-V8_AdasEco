import argparse
import cv2
import os
from ultralytics import YOLO

def load_model():
    # Load the pre-trained YOLOv8 model
    return YOLO('yolov8n.pt')

def print_detections(results):
    # List of classes to detect
    vehicle_types = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'van']
    detected_vehicles = []

    for result in results:
        for box in result.boxes.data:
            _, _, _, _, _, cls = box
            label = result.names[int(cls)]
            if label in vehicle_types:
                detected_vehicles.append(label)

    # Print summary of detections
    if detected_vehicles:
        print(f"Number of vehicles detected: {len(detected_vehicles)}")
        print("Types of vehicles detected:", set(detected_vehicles))
        print("List of all detected vehicles:", detected_vehicles)
    else:
        print("No vehicles detected")

def process_image(model, image_path):
    # Read the input image
    image = cv2.imread(image_path)
    # Perform object detection
    results = model(image)
    # Print detections
    print_detections(results)
    # Plot the results on the image
    annotated_image = results[0].plot()
    # Save the annotated image
    output_path = image_path.replace('.jpg', '_annotated.jpg')
    cv2.imwrite(output_path, annotated_image)
    print(f"Annotated image saved to {output_path}")

def process_video(model, video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create output video file path
    output_path = video_path.replace('.mp4', '_annotated.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)
        # Print detections
        print_detections(results)
        # Plot the results on the frame
        annotated_frame = results[0].plot()
        # Write the annotated frame to the output video
        out.write(annotated_frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Annotated video saved to {output_path}")

def main(input_path):
    # Load the YOLO model
    model = load_model()

    # Check if the input path is an image or a video
    if input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        process_image(model, input_path)
    elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
        process_video(model, input_path)
    else:
        print("Unsupported file format. Please provide an image or video file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection using YOLOv8.")
    parser.add_argument("input", help="Path to the input image or video file.")
    args = parser.parse_args()
    main(args.input)
