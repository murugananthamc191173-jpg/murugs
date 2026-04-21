import cv2
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs
from datetime import datetime
import time
import logging
import csv
import os
import queue
import threading
import torch  # For GPU check


# ---- CONFIGURE THESE ----
model_path = r"E:\assemblyvideo\Assembly_line_Latest_working_AUG\task_detection_ui\best_AUG_04.pt"
class_names = [
    "base_frame", "clamp", "fastener", "motor", "red_box",
    "screwdriver", "shaft", "wheel"
]
task_parts = {
    "fixing_clamp": 1,
    "fixing_motor": 3,
    "fixing_shaft": 6,
    "fixing_wheel": 7
}
TASK_ORDER = ["fixing_clamp", "fixing_motor", "fixing_shaft", "fixing_wheel"]
red_box_class_id = 4
max_red_box_area = 30000
min_conf = 0.20
nms_iou = 0.3
grace_frames = 5  # Grace period for state confirmation
csv_file = "task_times.csv"
log_file = "missing_actions.log"
low_fps_threshold = 15  # Warn if FPS drops below this


# Thread-safe queues
detection_queue = queue.Queue(maxsize=10)  # For detections
display_queue = queue.Queue(maxsize=10)    # New: For sending annotated frames back to main thread


# Background thread for processing and logging
class ProcessingThread(threading.Thread):
    def __init__(self, detection_queue, display_queue):
        super().__init__()
        self.detection_queue = detection_queue
        self.display_queue = display_queue
        self.task_states = {task: "NOT STARTED" for task in TASK_ORDER}
        self.state_counters = {task: 0 for task in TASK_ORDER}
        self.task_times = {task: {"start": None, "end": None} for task in TASK_ORDER}
        self.cycle_count = 0
        self.cycle_completed_flag = False  # Flag to prevent multiple increments
        self.log_queue = queue.Queue()
        self.should_stop = False
        self.csv_lock = threading.Lock()
        self.async_logger = AsyncLogger(self.log_queue, log_file, csv_file)
        self.async_logger.start()
        self.waiting_for_reset = False
        self.reset_start_time = None  # New: Track when waiting began
        self.reset_timeout = 30  # Seconds before forcing reset (adjust as needed)
        self.cycle_ready_to_log = False  # New: Flag to delay logging until full reset


    def run(self):
        while not self.should_stop:
            try:
                data = self.detection_queue.get(timeout=1)
                boxes, class_ids, confs, frame, fps = data['boxes'], data['class_ids'], data['confs'], data['frame'].copy(), data['fps']
                
                # Apply NMS (unchanged)
                unique_classes = np.unique(class_ids)
                for cls in unique_classes:
                    cls_idxs = [i for i, cid in enumerate(class_ids) if cid == cls]
                    if len(cls_idxs) > 1:
                        cls_boxes = [boxes[i] for i in cls_idxs]
                        cls_confs = [confs[i] for i in cls_idxs]
                        keep = self.nms(cls_boxes, cls_confs, iou_thresh=nms_iou)
                        remove = set(range(len(cls_boxes))) - set(keep)
                        for idx in sorted([cls_idxs[j] for j in remove], reverse=True):
                            boxes = np.delete(boxes, idx, axis=0)
                            class_ids = np.delete(class_ids, idx)
                            confs = np.delete(confs, idx)
                
                raw_status = self.detect_raw_status(boxes, class_ids)
                
                # Handle waiting for reset
                if self.waiting_for_reset:
                    current_time = datetime.now()
                    if self.reset_start_time is None:
                        self.reset_start_time = current_time
                    
                    # New: Update end times for returned objects during waiting
                    for task in TASK_ORDER:
                        if self.task_times[task]["start"] is not None and self.task_times[task]["end"] is None and not raw_status.get(task, False):
                            self.task_times[task]["end"] = current_time
                            self.log_queue.put({'type': 'log', 'message': f"Cycle {self.cycle_count - 1}: Set end time for {task} during reset."})
                    
                    if all(not raw_status.get(task, False) for task in TASK_ORDER):
                        # All objects back: Log the cycle now with accurate times
                        if self.cycle_ready_to_log:
                            csv_data = []
                            for task in TASK_ORDER:
                                start = self.task_times[task]["start"]
                                end = self.task_times[task]["end"]
                                duration = (end - start).total_seconds() if start and end else 0
                                csv_data.append([self.cycle_count - 1, task, start, end, duration])  # Use cycle_count - 1 since incremented early
                            self.log_queue.put({'type': 'csv', 'message': csv_data})
                            self.cycle_ready_to_log = False
                        
                        # Reset times and flags
                        for task in TASK_ORDER:
                            self.task_times[task] = {"start": None, "end": None}
                        self.waiting_for_reset = False
                        self.cycle_completed_flag = False
                        self.reset_start_time = None
                        self.log_queue.put({'type': 'log', 'message': f"Cycle {self.cycle_count}: Full reset confirmed. Ready for new cycle."})
                    elif (current_time - self.reset_start_time).total_seconds() > self.reset_timeout:
                        # Timeout: Log with current times (set end to now if pending)
                        if self.cycle_ready_to_log:
                            csv_data = []
                            for task in TASK_ORDER:
                                start = self.task_times[task]["start"]
                                end = self.task_times[task]["end"] or current_time  # Set to now if None
                                duration = (end - start).total_seconds() if start else 0
                                csv_data.append([self.cycle_count - 1, task, start, end, duration])
                            self.log_queue.put({'type': 'csv', 'message': csv_data})
                            self.cycle_ready_to_log = False
                        
                        # Reset and log
                        for task in TASK_ORDER:
                            self.task_states[task] = "NOT STARTED"
                            self.task_times[task] = {"start": None, "end": None}
                            self.state_counters[task] = 0
                        self.waiting_for_reset = False
                        self.cycle_completed_flag = False
                        self.reset_start_time = None
                        self.log_queue.put({'type': 'log', 'message': f"Cycle {self.cycle_count}: Reset timeout reached. Logged partial times and forced reset."})
                    else:
                        # During wait, force NOT STARTED but preserve times for logging
                        for task in TASK_ORDER:
                            self.task_states[task] = "NOT STARTED"
                            self.state_counters[task] = 0
                        missing_action, missing_task = False, None
                        annotated_frame = self.draw_boxes_and_tasks(frame, boxes, class_ids, confs, class_names, self.task_states, missing_action, missing_task, self.cycle_count, fps)
                        self.display_queue.put(annotated_frame.copy())
                        self.detection_queue.task_done()
                        continue
                
                # Normal updates (unchanged except for integration)
                self.task_states, self.task_times = self.update_task_states(raw_status, self.task_states, self.state_counters, self.task_times, TASK_ORDER)
                
                missing_action, missing_task = self.check_task_sequence(self.task_states, TASK_ORDER)
                if missing_action:
                    log_msg = f"Cycle {self.cycle_count}: Missing action detected: {missing_task} must be completed first. Current states: {self.task_states}"
                    self.log_queue.put({'type': 'log', 'message': log_msg})
                
                self.cycle_count, self.task_states, self.task_times = self.check_cycle_completion(self.task_states, TASK_ORDER, self.cycle_count, self.task_times, raw_status)
                
                if fps < low_fps_threshold:
                    self.log_queue.put({'type': 'log', 'message': f"Low FPS warning: {fps:.2f} in cycle {self.cycle_count}"})
                
                annotated_frame = self.draw_boxes_and_tasks(frame, boxes, class_ids, confs, class_names, self.task_states, missing_action, missing_task, self.cycle_count, fps)
                self.display_queue.put(annotated_frame.copy())
                self.detection_queue.task_done()
            except queue.Empty:
                pass
                
    def stop(self):
        self.should_stop = True
        self.async_logger.stop()
        self.async_logger.join()


    def box_area(self, box):
        return max(1, (box[2] - box[0]) * (box[3] - box[1]))


    def box_intersection(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        return max(0, xB - xA) * max(0, yB - yA)


    def is_80_percent_overlap(self, object_box, red_box):
        inter_area = self.box_intersection(object_box, red_box)
        obj_area = self.box_area(object_box)
        return inter_area / obj_area >= 0.7


    def nms(self, boxes, confidences, iou_thresh=0.3):
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        idxs = np.argsort(-confidences)
        keep = []
        while len(idxs) > 0:
            curr = idxs[0]
            keep.append(curr)
            if len(idxs) == 1:
                break
            rest = idxs[1:]
            ious = []
            for other in rest:
                inter = self.box_intersection(boxes[curr], boxes[other])
                union = self.box_area(boxes[curr]) + self.box_area(boxes[other]) - inter
                iou = inter / union if union > 0 else 0
                ious.append(iou)
            idxs = idxs[1:][np.array(ious) <= iou_thresh]
        return keep


    def detect_raw_status(self, boxes, class_ids):
        raw_status = {}
        valid_red_boxes = [
            boxes[i] for i in range(len(boxes))
            if class_ids[i] == red_box_class_id and self.box_area(boxes[i]) <= max_red_box_area
        ]
        for task, obj_cls in task_parts.items():
            obj_boxes = [boxes[i] for i in range(len(boxes)) if class_ids[i] == obj_cls]
            in_progress = all(
                not any(self.is_80_percent_overlap(obj_box, red_box) for red_box in valid_red_boxes)
                for obj_box in obj_boxes
            )
            raw_status[task] = in_progress
        return raw_status


    def update_task_states(self, raw_status, task_states, state_counters, task_times, task_order):
        for i, task in enumerate(task_order):
            raw = raw_status.get(task, False)
            if raw and state_counters[task] < grace_frames:
                state_counters[task] += 1
            elif not raw:
                state_counters[task] = 0
                if task_states[task] in ["IN PROGRESS", "COMPLETED"]:
                    task_states[task] = "NOT STARTED"
                    if task_times[task]["end"] is None:
                        task_times[task]["end"] = datetime.now()
            else:
                if task_states[task] == "NOT STARTED":
                    task_states[task] = "IN PROGRESS"
                    task_times[task]["start"] = datetime.now()
                    task_times[task]["end"] = None  # New: Reset end to prevent negatives on restart
                    if i > 0:
                        prev_task = task_order[i-1]
                        if task_states[prev_task] == "IN PROGRESS":
                            task_states[prev_task] = "COMPLETED"
                            task_times[prev_task]["end"] = datetime.now()
                state_counters[task] = grace_frames
        
        for i in range(len(task_order) - 1):
            current = task_order[i]
            next_task = task_order[i+1]
            if task_states[next_task] in ["IN PROGRESS", "COMPLETED"] and task_states[current] == "IN PROGRESS":
                task_states[current] = "COMPLETED"
                if not task_times[current]["end"]:
                    task_times[current]["end"] = datetime.now()
        
        return task_states, task_times
    
    def check_task_sequence(self, task_states, task_order):
        for i in range(len(task_order)):
            current_task = task_order[i]
            if task_states[current_task] != "COMPLETED":
                for later_task in task_order[i + 1:]:
                    if task_states[later_task] in ["IN PROGRESS", "COMPLETED"]:
                        return True, current_task
                break
        return False, None


    def check_cycle_completion(self, task_states, task_order, cycle_count, task_times, raw_status):
        if all(task_states[task] == "COMPLETED" for task in task_order[:-1]):
            if not self.cycle_completed_flag:
                cycle_count += 1
                self.cycle_completed_flag = True
                self.cycle_ready_to_log = True  # New: Set flag for delayed logging
                
                # Force state reset but preserve times for now
                for task in task_order:
                    task_states[task] = "NOT STARTED"
                self.waiting_for_reset = True
                self.reset_start_time = datetime.now()
                self.log_queue.put({'type': 'log', 'message': f"Cycle {cycle_count - 1} completed. Waiting for full reset to log times."})
            else:
                self.cycle_completed_flag = False
        return cycle_count, task_states, task_times




    def draw_boxes_and_tasks(self, frame, boxes, class_ids, confs, class_names, task_states, missing_action, missing_task, cycle_count, fps):
        for i, box in enumerate(boxes):
            xyxy = box.astype(int)
            # Clamp coordinates to frame size
            xyxy = np.clip(xyxy, 0, [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            class_id = class_ids[i]
            label = f"{class_names[class_id]} {confs[i]:.2f}"
            color = (0, 255, 0)
            if class_id == red_box_class_id:
                color = (0, 0, 255)
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(frame, label, (xyxy[0], max(xyxy[1] - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        y0 = 30
        for i, task in enumerate(TASK_ORDER):
            status = task_states[task]
            if status == "COMPLETED":
                text = f"{task}: COMPLETED"
                color = (0, 255, 0)
            elif status == "IN PROGRESS":
                text = f"{task}: IN PROGRESS"
                color = (0, 180, 0)
            else:
                text = f"{task}: NOT STARTED"
                color = (0, 0, 220)
            cv2.putText(frame, text, (10, y0 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(frame, f"Cycle: {cycle_count}", (10, y0 + len(TASK_ORDER) * 25 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if missing_action:
            message = f"Missing Action: {missing_task} must be completed first"
            cv2.putText(frame, message, (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        return frame


# AsyncLogger class (as before, but adjusted for multi-row CSV)
class AsyncLogger(threading.Thread):
    def __init__(self, queue, log_file, csv_file):
        super().__init__()
        self.queue = queue
        self.log_file = log_file
        self.csv_file = csv_file
        self.should_stop = False
        self.csv_lock = threading.Lock()


    def run(self):
        while not self.should_stop or not self.queue.empty():
            try:
                record = self.queue.get(timeout=1)
                if record['type'] == 'log':
                    with open(self.log_file, 'a') as f:
                        f.write(f"{datetime.now()} - {record['message']}\n")
                elif record['type'] == 'csv':
                    with self.csv_lock:
                        file_exists = os.path.isfile(self.csv_file) and os.path.getsize(self.csv_file) > 0
                        with open(self.csv_file, 'a', newline='') as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(["Cycle", "Task", "Start Time", "End Time", "Duration (s)"])
                            for row in record['message']:  # Handle multiple rows
                                writer.writerow(row)
                self.queue.task_done()
            except queue.Empty:
                pass


    def stop(self):
        self.should_stop = True


if __name__ == "__main__":
    try:
        model = YOLO(model_path)
        # Move to GPU if available (one-time)
        if torch.cuda.is_available():
            model.to('cuda')
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)


    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Error starting RealSense: {e}")
        exit(1)


    processing_thread = ProcessingThread(detection_queue, display_queue)
    processing_thread.start()


    prev_time = time.time()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data()).copy()  # Safe copy


            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time


            results = model(frame, conf=min_conf, iou=nms_iou)
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                detection_queue.put({
                    'boxes': boxes,
                    'class_ids': class_ids,
                    'confs': confs,
                    'frame': frame.copy(),  # Copy for queue
                    'fps': fps
                })


            # Check display queue and show in main thread (safe)
            try:
                annotated_frame = display_queue.get_nowait()
                # Diagnostics
                print("Frame shape:", annotated_frame.shape)
                print("Frame dtype:", annotated_frame.dtype)
                print("Frame min/max/mean:", np.min(annotated_frame), np.max(annotated_frame), np.mean(annotated_frame))
                if len(annotated_frame.shape) == 2 or annotated_frame.shape[2] == 1:
                    print("Warning: Frame is grayscale - converting to BGR")
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_GRAY2BGR)
                if annotated_frame.dtype != np.uint8:
                    print("Warning: Converting dtype to uint8")
                    annotated_frame = annotated_frame.astype(np.uint8)
                
                cv2.imshow("Task Detection", cv2.resize(annotated_frame, (640, 480)))  # Match config
                display_queue.task_done()
            except queue.Empty:
                pass  # No frame ready yet


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        processing_thread.stop()
        processing_thread.join()
        pipeline.stop()
        cv2.destroyAllWindows()
