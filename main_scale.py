import argparse  # For parsing command-line arguments
import cv2  # OpenCV library for image processing
from ultralytics import YOLO  # YOLO object detection model
import numpy as np  # For numerical operations
import concurrent.futures  # For parallel processing

def load_model():
    return YOLO('yolov8n.pt')  # Load the pre-trained YOLOv8 nano (tiny) model

def estimate_dimensions(box, image_height, image_width):
    # Estimate dimensions of detected vehicle
    width = box[2] - box[0]  # Calculate width of bounding box
    height = box[3] - box[1]  # Calculate height of bounding box
    # Assume typical car dimensions and scale accordingly
    length = width / image_width * 4.5
    vehicle_width = width / image_width * 1.8
    vehicle_height = height / image_height * 1.5
    return f"{length:.2f}m x {vehicle_width:.2f}m x {vehicle_height:.2f}m"  # Return formatted string of dimensions

def predict_trajectory(prev_centers, current_center):
    if len(prev_centers) < 2:  # If not enough previous positions, return current center
        return current_center
    velocity = np.array(current_center) - np.array(prev_centers[-1])  # Calculate velocity
    next_point = np.array(current_center) + velocity  # Predict next point
    return tuple(map(int, next_point))  # Return predicted point as integer tuple

def process_frame(model, frame):
    results = model(frame)  # Run YOLO model on the frame
    annotated_frame = results[0].plot()  # Plot the results on the frame
    detected_vehicles = {}  # Dictionary to store detected vehicles

    for box in results[0].boxes.data:  # Loop through detected objects
        x1, y1, x2, y2, conf, cls = box  # Unpack box data
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert coordinates to integers
        center = ((x1 + x2) // 2, (y1 + y2) // 2)  # Calculate center of the box

        # Estimate and draw dimensions
        dimensions = estimate_dimensions((x1, y1, x2, y2), frame.shape[0], frame.shape[1])
        cv2.putText(annotated_frame, dimensions, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Red text at bottom

        # Predict and draw trajectory
        if cls not in detected_vehicles:
            detected_vehicles[cls] = []
        detected_vehicles[cls].append(center)
        if len(detected_vehicles[cls]) > 10:  # Keep only last 10 positions
            detected_vehicles[cls].pop(0)

        if len(detected_vehicles[cls]) >= 2:
            next_point = predict_trajectory(detected_vehicles[cls][:-1], center)
            cv2.line(annotated_frame, center, next_point, (0, 255, 0), 2)

    return annotated_frame  # Return the annotated frame

def process_video(model, video_path, skip_frames=5):
    cap = cv2.VideoCapture(video_path)  # Open the video file
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get video width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get video height
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get video fps

    output_path = video_path.replace('.mp4', '_annotated.mp4')  # Create output file path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))  # Create VideoWriter object

    frame_count = 0  # Counter for processed frames

    with concurrent.futures.ThreadPoolExecutor() as executor:  # Use thread pool for parallel processing
        futures = []  # List to store futures

        while cap.isOpened():  # Loop through video frames
            ret, frame = cap.read()  # Read a frame
            if not ret:  # If frame reading was unsuccessful, break the loop
                break

            if frame_count % skip_frames == 0:  # Process only every nth frame
                futures.append(executor.submit(process_frame, model, frame))  # Submit frame for processing

            frame_count += 1  # Increment frame counter

        for future in concurrent.futures.as_completed(futures):  # As each frame is processed
            annotated_frame = future.result()  # Get the result
            out.write(annotated_frame)  # Write the annotated frame to output video

        if frame_count % 100 == 0:  # Print progress every 100 frames
            print(f"Processed {frame_count} frames")

    cap.release()  # Release the video capture object
    out.release()  # Release the video writer object
    cv2.destroyAllWindows()  # Close all OpenCV windows
    print(f"Annotated video saved to {output_path}")  # Print output file path

def count_vehicles(results):
    if results is None:  # Check if results is None
        return {}

    vehicle_types = [
        'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'van', 'trailer',
        'ambulance', 'fire truck', 'police car', 'taxi', 'pickup truck', 'suv',
        'minivan', 'sports car', 'convertible', 'limousine', 'rv', 'tractor',
        'forklift', 'atv', 'golf cart', 'snowmobile'
    ]  # Comprehensive list of vehicle types to count
    counts = {type: 0 for type in vehicle_types}  # Initialize counts

    for result in results:
        for cls in result.boxes.cls:
            label = result.names[int(cls)]  # Get the label for the class
            if label in vehicle_types:
                counts[label] += 1  # Increment count for the vehicle type

    return counts  # Return the counts

def main(input_path):
    model = load_model()  # Load the YOLO model

    if input_path.endswith(('.mp4', '.avi')):  # Check if input is a video file
        process_video(model, input_path)  # Process the video
    else:
        print("Unsupported file format. Please use a video file.")  # Print error for unsupported formats
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle detection and counting using YOLOv8.")
    parser.add_argument("input", help="Path to the input video file.")
    args = parser.parse_args()  # Parse command-line arguments
    main(args.input)  # Call main function with input argument
