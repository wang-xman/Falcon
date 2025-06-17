#!/usr/bin/env python3
"""
Multi-Chip Multi-Modal Multi-Stream Inference - Minimal Working Example

Preprocess, postprocess, and inference.
"""
import argparse
import os
import cv2
import queue
from pathlib import Path
import threading
import cv2
from loguru import logger

from object_detection_utils import ObjectDetectionUtils, load_images_opencv, \
        load_input_images, divide_list_to_batches, validate_images
from async_inference_engine import HailoAsyncInference

CAMERA_CAP_WIDTH = 640
CAMERA_CAP_HEIGHT = 640


def parse_args() -> argparse.Namespace:
    """Initialize argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Detection Example", allow_abbrev=False)
    parser.add_argument(
        "--input_device", 
        help="Input device such as camera, video or image",
        default="camera"
    )
    parser.add_argument(
        "--camera_index", 
        default=0,
        type=int,
        help="Camera index in for example /dev/video0 is 0."
    )
    parser.add_argument(
        "--hailo_device", 
        default=0,
        type=int,
        help="Index of Hailo devices. Index can be found by hailortcli scan."
    )
    parser.add_argument(
        "--save_output_stream", 
        default=False,
        type=bool,
        help="Save output stream if True. Default to False."
    )
    parser.add_argument(
        "--network_path",
        default="./yolov8m.hef",
        help="Path to compiled Hailo Execution File (HEF) file."
    )
    parser.add_argument(
        "--label_file",
        default="./coco.txt",
        help="Path to label file. Default is the coco."
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size for inferencing. Default to 8."
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.network_path):
        raise FileNotFoundError(f"Network file not found: {args.network_path}")
    if not os.path.exists(args.label_file):
        raise FileNotFoundError(f"Labels file not found: {args.label_file}")

    return args


def preprocess(
        cap: cv2.VideoCapture,
        batch_size: int,
        input_queue: queue.Queue,
        width: int,
        height: int,
        utils: ObjectDetectionUtils) -> None:
    """Process frames from the camera stream and enqueue them.

    Args:
        batch_size (int): Number of images per batch.
        input_queue (queue.Queue): Queue for input images.
        width (int): Model input width.
        height (int): Model input height.
    """
    frames = []
    processed_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Call utils object to preprocess a frame
        processed_frame = utils.preprocess(processed_frame, width, height)
        processed_frames.append(processed_frame)

        if len(frames) == batch_size:
            input_queue.put((frames, processed_frames))
            processed_frames, frames = [], []


def postprocess(
        output_queue: queue.Queue,
        cap: cv2.VideoCapture,
        output_window: str,
        save_stream_output: bool,
        utils: ObjectDetectionUtils) -> None:
    """Process and visualize the output results.

    Args:
        output_queue (queue.Queue): Queue for output results.
        camera (bool): Flag indicating if the input is from a camera.
        save_stream_output (bool): Flag indicating if the camera output should be saved.
        utils (ObjectDetectionUtils): Utility class for object detection visualization.
    """
    image_id = 0
    out = None
    output_path = Path('output')
    if cap is not None:
        if save_stream_output:
            output_path.mkdir(exist_ok=True)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
             # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # Save the output video in the output path
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:  # If FPS is not available, set a default value
                fps = 20.0
            out = cv2.VideoWriter(str(output_path / 'output_video.avi'), fourcc, fps, (frame_width, frame_height))

    if (cap is None):
        # Create output directory if it doesn't exist
        output_path.mkdir(exist_ok=True)

    while True:
        result = output_queue.get()
        if result is None:
            break  # Exit the loop if sentinel value is received

        original_frame, infer_results = result

        # Deals with the expanded results from hailort versions < 4.19.0
        if len(infer_results) == 1:
            infer_results = infer_results[0]
            print(infer_results)

        detections = utils.extract_detections(infer_results)

        frame_with_detections = utils.draw_detections(
            detections, original_frame,
        )
        
        if cap is not None:
            # Display output
            cv2.imshow(output_window, frame_with_detections)
            if save_stream_output:
                out.write(frame_with_detections)
        else:
            cv2.imwrite(str(output_path / f"output_{image_id}.png"), frame_with_detections)

        # Wait for key press "q"
        image_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Close the window and release the camera
            if save_stream_output:
                out.release()  # Release the VideoWriter object
            cap.release()
            cv2.destroyAllWindows()
            break

    if cap is not None and save_stream_output:
            out.release()  # Release the VideoWriter object
    output_queue.task_done()  # Indicate that processing is complete


def infer(
        input,
        camera_index,
        device_index,
        save_stream_output: bool,
        net_path: str,
        labels_path: str,
        batch_size: int) -> None:
    """Initialize queues, HailoAsyncInference instance, and run the inference.

    Args:
        images (List[Image.Image]): List of images to process.
        net_path (str): Path to the HEF model file.
        labels_path (str): Path to a text file containing labels.
        batch_size (int): Number of images per batch.
        output_path (Path): Path to save the output images.
    """
    det_utils = ObjectDetectionUtils(labels_path)
    
    cap = None
    images = []
    if input == "camera":
        # Capture the designated camera
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CAP_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CAP_HEIGHT)
    elif any(input.lower().endswith(suffix) for suffix in ['.mp4', '.avi', '.mov', '.mkv']):
        cap = cv2.VideoCapture(input)
    else:
        images = load_images_opencv(input)

        # Validate images
        try:
            validate_images(images, batch_size)
        except ValueError as e:
            logger.error(e)
            return

    input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    # Added device_index as the 4th argument
    hailo_inference = HailoAsyncInference(
        net_path, input_queue, output_queue, device_index, batch_size, send_original_frame=True
    )
    height, width, _ = hailo_inference.get_input_shape()

    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(cap, batch_size, input_queue, width, height, det_utils)
    )
    postprocess_thread = threading.Thread(
        target=postprocess,
        args=(output_queue, cap, "Output"+str(camera_index), save_stream_output, det_utils)
    )

    preprocess_thread.start()
    postprocess_thread.start()

    hailo_inference.run()
    
    preprocess_thread.join()
    output_queue.put(None)  # Signal process thread to exit
    postprocess_thread.join()

    logger.info('Inference was successful!')


if __name__ == '__main__':
    """Example: Run inference using command line

    Inferencing using camera 2, Hailo chip 1, YOLOv7, and batch size 8

    $ python3 example.py --input_device camera --camera_index 2 --hailo_device 1 --network_path ./yolov7.hef

    Inferencing using video clip, here camera index is irrelevant

    $ python3 example.py --input_device walkers.mp4 --camera_index 0 --hailo_device 1 --network_path ./yolov8m.hef --batch_size 1
    """
    args = parse_args()
    # inferencing on livestream
    infer(args.input_device, args.camera_index, args.hailo_device,
          args.save_output_stream, args.network_path, args.label_file, args.batch_size)