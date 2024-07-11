import threading
import cv2
from queue import Queue
from model import YOLO  # Adjusted import statement
from plotting import draw_boxes  # Adjusted import statement
from torch_utils import select_device, load_model  # Adjusted import statement
from files import read_video, save_video  # Adjusted import statement
from data.dataset import Dataset
from data.loaders import DataLoader
from annotator import Annotator  # Adjusted import statement
from data.utils import resize_image, normalize_data

def video_thread(video_source, frame_queue, stop_event):
    video_frames = read_video(video_source)
    for frame in video_frames:
        if stop_event.is_set():
            break;
        frame_queue.put(frame)
    stop_event.set()  # Signal other threads to stop

def detection_thread(frame_queue, result_queue, model, stop_event):
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=1)
            results = model(frame)
            result_queue.put((frame, results))
        except queue.Empty:
            continue

def processing_thread(result_queue, classes_to_count, class_names, output_path, stop_event):
    while not stop_event.is_set() or not result_queue.empty():
        try:
            frame, results = result_queue.get(timeout=1)
            counts = {cls: 0 for cls in classes_to_count}
            for result in results:
                cls_id = result[-1]
                if cls_id in classes_to_count:
                    counts[cls_id] += 1
            annotated_frame = Annotator().annotate(frame, results)
            save_video(output_path, annotated_frame)
        except queue.Empty:
            continue

if __name__ == "__main__":
    frame_queue = Queue()
    result_queue = Queue()
    classes_to_count = [0, 2]
    class_names = {0: 'person', 2: 'car'}
    video_source = 'data/testvideo.mp4'
    output_path = 'data/output_video.avi'
    stop_event = threading.Event()
    device = select_device()
    model = load_model('ultralytics/yolov8n.pt', device)

    threads = [
        threading.Thread(target=video_thread, args=(video_source, frame_queue, stop_event)),
        threading.Thread(target=detection_thread, args=(frame_queue, result_queue, model, stop_event)),
        threading.Thread(target=processing_thread, args=(result_queue, classes_to_count, class_names, output_path, stop_event)),
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
