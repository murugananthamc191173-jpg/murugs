# main_detection_ui.py
import sys

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMessageBox

import cv2
import numpy as np
import pyrealsense2 as rs
from datetime import datetime
import time
from ultralytics import YOLO
import torch
import queue
import threading

from lightweight_ui import LightweightUI
from run_script import ProcessingThread, AsyncLogger  # Assuming these are defined in task_detection_v3_5.py

class DetectionThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    metrics_ready = pyqtSignal(int, float)
    tasks_ready = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        print("Initializing DetectionThread")
        # Initialize variables
        self.running = False
        self.pipeline = None
        self.model = None
        self.processing_thread = None
        self.detection_queue = queue.Queue(maxsize=10)
        self.display_queue = queue.Queue(maxsize=10)

    def initialize_camera(self):
        """Initialize the RealSense camera"""
        try:
            print("Setting up camera...")
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(config)
            print("Camera initialized successfully")
            return True
        except Exception as e:
            self.error_signal.emit(f"Camera initialization failed: {str(e)}")
            print(f"Camera error: {e}")
            return False

    def initialize_model(self):
        """Initialize the YOLO model"""
        try:
            print("Loading YOLO model...")
            model_path = r"E:\assemblyvideo\Assembly_line_Latest_working_AUG\task_detection_ui\best_AUG_04.pt"
            self.model = YOLO(model_path)
            if torch.cuda.is_available():
                print("Using GPU")
                self.model.to('cuda')
            print("Model loaded successfully")
            return True
        except Exception as e:
            self.error_signal.emit(f"Model initialization failed: {str(e)}")
            print(f"Model error: {e}")
            return False

    def initialize_processing(self):
        """Initialize the processing thread"""
        try:
            print("Setting up processing thread...")
            self.processing_thread = ProcessingThread(self.detection_queue, self.display_queue)
            print("Processing thread ready")
            return True
        except Exception as e:
            self.error_signal.emit(f"Processing thread initialization failed: {str(e)}")
            print(f"Processing thread error: {e}")
            return False

    def start_detection(self):
        """Start the detection process"""
        print("Starting detection...")
        if self.running:
            print("Detection already running")
            return
        # Initialize components
        if not all([
            self.initialize_camera(),
            self.initialize_model(),
            self.initialize_processing()
        ]):
            print("Initialization failed")
            return
        self.running = True
        self.processing_thread.start()
        super().start()
        print("Detection started")

    def stop_detection(self):
        """Stop the detection process"""
        print("Stopping detection...")
        self.running = False
        self.cleanup()
        print("Detection stopped")

    def run(self):
        """Main detection loop"""
        print("Detection thread running")
        try:
            prev_time = time.time()
            while self.running:
                # Get frame from camera
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                # Process frame
                frame = np.asanyarray(color_frame.get_data()).copy()
                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                # Run detection
                results = self.model(frame, conf=0.20, iou=0.3)
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    # Process detection results
                    boxes = result.boxes.xyxy.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    confs = result.boxes.conf.cpu().numpy()
                    # Queue detection results
                    self.detection_queue.put({
                        'boxes': boxes,
                        'class_ids': class_ids,
                        'confs': confs,
                        'frame': frame.copy(),
                        'fps': fps
                    })
                # Update UI with processed frame
                try:
                    annotated_frame = self.display_queue.get_nowait()
                    self.frame_ready.emit(annotated_frame)
                    self.metrics_ready.emit(
                        self.processing_thread.cycle_count,
                        fps
                    )
                    self.tasks_ready.emit(self.processing_thread.task_states)
                    self.display_queue.task_done()
                except queue.Empty:
                    # Show raw frame if no processed frame is available
                    self.frame_ready.emit(frame)
                    self.metrics_ready.emit(0, fps)
        except Exception as e:
            self.error_signal.emit(f"Detection loop error: {str(e)}")
            print(f"Detection loop error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up resources...")
        try:
            if self.processing_thread is not None:
                self.processing_thread.stop()
                self.processing_thread.join()
                self.processing_thread = None
            if self.pipeline is not None:
                self.pipeline.stop()
                self.pipeline = None
            self.model = None
            print("Cleanup complete")
        except Exception as e:
            self.error_signal.emit(f"Cleanup error: {str(e)}")
            print(f"Cleanup error: {e}")

class MainApplication:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.ui = None
        self.detection_thread = None

    def setup(self):
        """Setup the application"""
        try:
            # Create UI
            print("Creating UI...")
            self.ui = LightweightUI()
            # Create detection thread
            print("Creating detection thread...")
            self.detection_thread = DetectionThread()
            # Connect signals
            print("Connecting signals...")
            self.ui.start_signal.connect(self.detection_thread.start_detection)
            self.ui.stop_signal.connect(self.detection_thread.stop_detection)
            self.detection_thread.frame_ready.connect(self.ui.update_camera_feed)
            self.detection_thread.metrics_ready.connect(self.ui.update_metrics)
            self.detection_thread.tasks_ready.connect(self.ui.update_task_status)
            self.detection_thread.error_signal.connect(self.show_error)
            # Show UI
            self.ui.show()
            print("Setup complete")
            return True
        except Exception as e:
            print(f"Setup error: {e}")
            QMessageBox.critical(None, "Setup Error", str(e))
            return False

    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(None, "Error", message)

    def run(self):
        """Run the application"""
        if self.setup():
            return self.app.exec()
        return 1

def main():
    app = MainApplication()
    return app.run()

if __name__ == "__main__":
    sys.exit(main())
